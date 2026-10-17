"""Training loop for D-MPNN property prediction with early stopping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from nexus.models.autograd import Tensor
from nexus.models.featurize import MolGraph, featurize_batch
from nexus.models.metrics import multitask_score
from nexus.models.mpnn import DMPNN
from nexus.models.nn import (
    Adam,
    CosineSchedule,
    binary_cross_entropy_with_logits,
    mse_loss,
)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden: int = 300
    depth: int = 3
    dropout: float = 0.0
    readout: str = "mean"
    ffn_layers: int = 2
    patience: int = 15
    grad_clip: float = 5.0
    warmup_fraction: float = 0.1
    seed: int = 0
    standardize_targets: bool = True
    verbose: bool = False


@dataclass
class TrainResult:
    model: DMPNN
    best_epoch: int
    best_valid: float
    history: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None


def _precompute_graphs(mols, batch_size: int, order: np.ndarray) -> list[tuple[np.ndarray, MolGraph]]:
    out = []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        out.append((idx, featurize_batch([mols[i] for i in idx])))
    return out


def predict(model: DMPNN, mols, batch_size: int = 100,
            extra: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(mols), batch_size):
        chunk = mols[start:start + batch_size]
        graph = featurize_batch(chunk)
        extra_chunk = None if extra is None else extra[start:start + batch_size]
        preds.append(model(graph, extra_chunk).data)
    model.train()
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 1))


def train_model(
    train_mols, Y_train: np.ndarray,
    valid_mols, Y_valid: np.ndarray,
    task_type: str,
    config: TrainConfig = TrainConfig(),
    eval_metric: str | None = None,
    extra_train: np.ndarray | None = None,
    extra_valid: np.ndarray | None = None,
) -> TrainResult:
    """Train a D-MPNN, keeping the parameters with the best validation score."""
    Y_train = np.atleast_2d(Y_train.T).T if Y_train.ndim == 1 else Y_train
    Y_valid = np.atleast_2d(Y_valid.T).T if Y_valid.ndim == 1 else Y_valid
    n_tasks = Y_train.shape[1]
    classification = task_type == "classification"
    metric = eval_metric or ("roc_auc" if classification else "rmse")
    higher_is_better = metric in ("roc_auc", "prc_auc", "r2", "accuracy",
                                  "pearson", "spearman")

    y_mean = y_std = None
    target = Y_train
    if not classification and config.standardize_targets:
        y_mean = np.nanmean(Y_train, axis=0)
        y_std = np.nanstd(Y_train, axis=0)
        y_std[y_std == 0] = 1.0
        target = (Y_train - y_mean) / y_std

    model = DMPNN(hidden=config.hidden, depth=config.depth, n_tasks=n_tasks,
                  dropout=config.dropout, readout=config.readout,
                  ffn_layers=config.ffn_layers,
                  extra_features=0 if extra_train is None else extra_train.shape[1],
                  seed=config.seed)
    optimizer = Adam(model.parameters(), lr=config.lr,
                     weight_decay=config.weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(train_mols) / config.batch_size)))
    schedule = CosineSchedule(optimizer, config.epochs * steps_per_epoch,
                              warmup=int(config.warmup_fraction * config.epochs * steps_per_epoch))

    rng = np.random.default_rng(config.seed)
    best_state, best_valid, best_epoch = None, -np.inf, 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(train_mols))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            idx = order[start:start + config.batch_size]
            graph = featurize_batch([train_mols[i] for i in idx])
            y = target[idx]
            mask = np.isfinite(y)
            if not mask.any():
                continue
            y_filled = np.where(mask, y, 0.0)
            extra_batch = None if extra_train is None else extra_train[idx]
            optimizer.zero_grad()
            out = model(graph, extra_batch)
            if classification:
                loss = binary_cross_entropy_with_logits(out, y_filled, mask=mask)
            else:
                loss = mse_loss(out, y_filled, mask=mask)
            loss.backward()
            optimizer.clip_grad_norm(config.grad_clip)
            optimizer.step()
            schedule.step()
            epoch_loss += float(loss.data)
            n_batches += 1

        raw = predict(model, valid_mols, config.batch_size, extra_valid)
        pred = 1.0 / (1.0 + np.exp(-raw)) if classification else raw
        if y_mean is not None:
            pred = pred * y_std + y_mean
        score = multitask_score(Y_valid, pred, metric)
        signed = score if higher_is_better else -score
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1),
                        f"valid_{metric}": score, "lr": optimizer.lr})
        if config.verbose:
            print(f"  epoch {epoch:3d} loss {epoch_loss / max(n_batches,1):.4f} "
                  f"valid {metric} {score:.4f}", flush=True)
        if np.isfinite(signed) and signed > best_valid:
            best_valid, best_epoch = signed, epoch
            best_state = model.state_dict()
        elif epoch - best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_epoch=best_epoch,
                       best_valid=best_valid if higher_is_better else -best_valid,
                       history=history, train_seconds=time.time() - t0,
                       y_mean=y_mean, y_std=y_std)


def predict_calibrated(result: TrainResult, mols, task_type: str,
                       batch_size: int = 100, extra: np.ndarray | None = None) -> np.ndarray:
    """Predictions on the original target scale (probabilities for classification)."""
    raw = predict(result.model, mols, batch_size, extra)
    if task_type == "classification":
        return 1.0 / (1.0 + np.exp(-raw))
    if result.y_mean is not None:
        return raw * result.y_std + result.y_mean
    return raw
