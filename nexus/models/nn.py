"""Neural-network building blocks on the NEXUS autograd engine."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from nexus.models.autograd import Tensor, concat


class Module:
    """Base class: tracks parameters and child modules by attribute."""

    def parameters(self) -> list[Tensor]:
        seen: dict[int, Tensor] = {}
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                seen[id(value)] = value
            elif isinstance(value, Module):
                for p in value.parameters():
                    seen[id(p)] = p
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        for p in item.parameters():
                            seen[id(p)] = p
                    elif isinstance(item, Tensor) and item.requires_grad:
                        seen[id(item)] = item
        return list(seen.values())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def n_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def train(self) -> "Module":
        self._set_training(True)
        return self

    def eval(self) -> "Module":
        self._set_training(False)
        return self

    def _set_training(self, flag: bool) -> None:
        self.training = flag
        for value in vars(self).values():
            if isinstance(value, Module):
                value._set_training(flag)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item._set_training(flag)

    def state_dict(self) -> list[np.ndarray]:
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, state: list[np.ndarray]) -> None:
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError(f"expected {len(params)} arrays, got {len(state)}")
        for p, arr in zip(params, state):
            if p.shape != arr.shape:
                raise ValueError(f"shape mismatch: {p.shape} vs {arr.shape}")
            p.data = np.array(arr, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.state_dict()))

    def load(self, path: str | Path) -> "Module":
        self.load_state_dict(pickle.loads(Path(path).read_bytes()))
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """Fully connected layer with Glorot-uniform initialisation."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (n_in + n_out))
        self.weight = Tensor(rng.uniform(-limit, limit, (n_in, n_out)), requires_grad=True)
        self.bias = Tensor(np.zeros(n_out), requires_grad=True) if bias else None
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, n_items: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weight = Tensor(rng.normal(0, 0.1, (n_items, dim)), requires_grad=True)
        self.training = True

    def forward(self, index: np.ndarray) -> Tensor:
        return self.weight[np.asarray(index, dtype=np.int64)]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centred = x - mu
        var = (centred * centred).mean(axis=-1, keepdims=True)
        return centred / (var + self.eps).sqrt() * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p: float = 0.1, seed: int = 0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0:
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * Tensor(mask)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Module):
    def __init__(self, sizes: list[int], activation: str = "relu",
                 dropout: float = 0.0, layer_norm: bool = False, seed: int = 0):
        self.linears = [Linear(a, b, seed=seed + i)
                        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        self.norms = [LayerNorm(b) for b in sizes[1:-1]] if layer_norm else []
        self.dropouts = [Dropout(dropout, seed=seed + 100 + i)
                         for i in range(len(self.linears) - 1)] if dropout > 0 else []
        self.activation = activation
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        last = len(self.linears) - 1
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < last:
                if self.norms:
                    x = self.norms[i](x)
                x = getattr(x, self.activation)()
                if self.dropouts:
                    x = self.dropouts[i](x)
        return x


# ------------------------------------------------------------------ optimisers

class Optimizer:
    def __init__(self, parameters: Iterable[Tensor], lr: float):
        self.parameters = list(parameters)
        self.lr = lr
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def clip_grad_norm(self, max_norm: float) -> float:
        total = math.sqrt(sum(float((p.grad ** 2).sum())
                              for p in self.parameters if p.grad is not None))
        if total > max_norm and total > 0:
            scale = max_norm / total
            for p in self.parameters:
                if p.grad is not None:
                    p.grad *= scale
        return total

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.step_count += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]


class Adam(Optimizer):
    """Adam (Kingma & Ba 2015); `decoupled=True` gives AdamW."""

    def __init__(self, parameters, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, decoupled: bool = False):
        super().__init__(parameters, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** t)
            v_hat = self.v[i] / (1 - self.b2 ** t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay and self.decoupled:
                update = update + self.lr * self.weight_decay * p.data
            p.data -= update


def AdamW(parameters, lr: float = 1e-3, weight_decay: float = 0.01, **kwargs) -> Adam:
    return Adam(parameters, lr=lr, weight_decay=weight_decay, decoupled=True, **kwargs)


class CosineSchedule:
    """Cosine learning-rate decay with linear warmup."""

    def __init__(self, optimizer: Optimizer, total_steps: int, warmup: int = 0,
                 min_lr_fraction: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.total_steps = max(total_steps, 1)
        self.warmup = warmup
        self.min_fraction = min_lr_fraction
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.warmup and self.step_count <= self.warmup:
            factor = self.step_count / self.warmup
        else:
            progress = ((self.step_count - self.warmup)
                        / max(self.total_steps - self.warmup, 1))
            progress = min(max(progress, 0.0), 1.0)
            factor = self.min_fraction + (1 - self.min_fraction) * 0.5 * (
                1 + math.cos(math.pi * progress))
        self.optimizer.lr = self.base_lr * factor
        return self.optimizer.lr


# ---------------------------------------------------------------------- losses

def mse_loss(pred: Tensor, target: np.ndarray, mask: np.ndarray | None = None) -> Tensor:
    diff = pred - Tensor(np.asarray(target, dtype=np.float64).reshape(pred.shape))
    sq = diff * diff
    if mask is None:
        return sq.mean()
    m = Tensor(np.asarray(mask, dtype=np.float64).reshape(pred.shape))
    return (sq * m).sum() * (1.0 / max(float(np.sum(mask)), 1.0))


def l1_loss(pred: Tensor, target: np.ndarray) -> Tensor:
    return (pred - Tensor(np.asarray(target).reshape(pred.shape))).abs().mean()


def huber_loss(pred: Tensor, target: np.ndarray, delta: float = 1.0) -> Tensor:
    diff = pred - Tensor(np.asarray(target).reshape(pred.shape))
    a = diff.abs()
    quadratic = a.clip(0.0, delta)
    linear = a - quadratic
    return (quadratic * quadratic * 0.5 + linear * delta).mean()


def binary_cross_entropy_with_logits(
    logits: Tensor, target: np.ndarray, mask: np.ndarray | None = None,
    pos_weight: float = 1.0,
) -> Tensor:
    """Numerically stable BCE on logits, with optional per-element masking."""
    y = np.asarray(target, dtype=np.float64).reshape(logits.shape)
    # softplus(-|z|) + max(z, 0) - z*y  is the stable form of the BCE loss.
    stable = (-logits.abs()).softplus() + logits.clip(0.0, np.inf) - logits * Tensor(y)
    if pos_weight != 1.0:
        weights = Tensor(np.where(y > 0.5, pos_weight, 1.0))
        stable = stable * weights
    if mask is None:
        return stable.mean()
    m = np.asarray(mask, dtype=np.float64).reshape(logits.shape)
    return (stable * Tensor(m)).sum() * (1.0 / max(float(m.sum()), 1.0))


def cross_entropy(logits: Tensor, target: np.ndarray) -> Tensor:
    """Multiclass cross entropy from integer class labels."""
    log_probs = logits.log_softmax(axis=-1)
    idx = np.asarray(target, dtype=np.int64)
    rows = np.arange(len(idx))
    return -log_probs[rows, idx].mean()


def kl_divergence_standard_normal(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)), summed over features and averaged over rows."""
    return (((mu * mu) + log_var.exp() - 1.0 - log_var) * 0.5).sum(axis=-1).mean()
