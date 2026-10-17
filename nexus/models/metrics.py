"""Evaluation metrics for molecular property prediction."""

from __future__ import annotations

import numpy as np


def _mask(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(y) & np.isfinite(p)
    return y[keep], p[keep]


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) identity."""
    y, scores = _mask(np.asarray(y, float), np.asarray(scores, float))
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so tied scores contribute 0.5 each.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (PRC-AUC), step interpolation."""
    y, scores = _mask(np.asarray(y, float), np.asarray(scores, float))
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def accuracy(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    y, scores = _mask(np.asarray(y, float), np.asarray(scores, float))
    return float(np.mean((scores >= threshold) == (y == 1)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _mask(np.asarray(y, float), np.asarray(pred, float))
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _mask(np.asarray(y, float), np.asarray(pred, float))
    return float(np.mean(np.abs(y - pred)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _mask(np.asarray(y, float), np.asarray(pred, float))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _mask(np.asarray(y, float), np.asarray(pred, float))
    if len(y) < 2 or y.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _mask(np.asarray(y, float), np.asarray(pred, float))
    if len(y) < 2:
        return float("nan")
    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        return r
    return pearson(rank(y), rank(pred))


def multitask_score(Y: np.ndarray, P: np.ndarray, metric: str) -> float:
    """Average a metric over tasks, skipping tasks with no usable labels."""
    fn = {"roc_auc": roc_auc, "prc_auc": average_precision, "rmse": rmse,
          "mae": mae, "r2": r2, "accuracy": accuracy, "pearson": pearson,
          "spearman": spearman}[metric]
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    P = np.atleast_2d(P.T).T if P.ndim == 1 else P
    scores = []
    for t in range(Y.shape[1]):
        keep = np.isfinite(Y[:, t])
        if keep.sum() == 0:
            continue
        value = fn(Y[keep, t], P[keep, t])
        if np.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")
