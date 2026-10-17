"""Reverse-mode automatic differentiation over NumPy arrays.

A minimal tape-based engine: every operation records how to push gradients to
its inputs, and `Tensor.backward()` walks the graph in reverse topological
order. Enough to express message-passing networks, diffusion models and
variational objectives without an external deep-learning framework.

Every operation here is checked against finite differences in
tests/test_autograd.py.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

ArrayLike = "np.ndarray | float | int | Tensor"


def _noop() -> None:
    """Placeholder installed on freed graph nodes."""


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a gradient back to `shape`, undoing NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    # Leading axes that broadcasting added.
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    # Axes that were size 1 and got expanded.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that records operations for backpropagation."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children: tuple = (),
                 _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Tensor, ...] = _children
        self._op = _op

    # --------------------------------------------------------------- plumbing

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, g: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.array(g, dtype=np.float64)
        else:
            self.grad += g

    def backward(self, gradient: np.ndarray | None = None,
                 retain_graph: bool = False) -> None:
        """Backpropagate from this tensor (a scalar unless `gradient` is given).

        Each operation's backward closure captures its own output tensor, so the
        graph contains reference cycles that Python frees only on a garbage
        collection pass. Unless `retain_graph` is set, the graph is released
        explicitly once gradients are computed, which keeps memory flat across
        long training runs instead of growing until the process is killed.
        """
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar tensor needs an explicit gradient")
            gradient = np.ones_like(self.data)
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        self._accumulate(np.asarray(gradient, dtype=np.float64))
        for node in reversed(topo):
            node._backward()
        if not retain_graph:
            for node in topo:
                node._backward = _noop
                node._prev = ()

    # ------------------------------------------------------------- arithmetic

    def _wrap(self, other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other), "add")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other), "mul")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, self.requires_grad, (self,), "pow")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * power * self.data ** (power - 1))
        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    def __truediv__(self, other) -> "Tensor":
        return self * (self._wrap(other) ** -1.0)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other) -> "Tensor":
        return self._wrap(other) * (self ** -1.0)

    def __matmul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accumulate(_unbroadcast(g, self.shape))
            if other.requires_grad:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accumulate(_unbroadcast(g, other.shape))
        out._backward = _backward
        return out

    # ------------------------------------------------------------- reductions

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        values = self.data.max(axis=axis, keepdims=True)
        out = Tensor(values if keepdims else values.squeeze(axis),
                     self.requires_grad, (self,), "max")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            mask = (self.data == values)
            # Split the gradient evenly between tied maxima.
            self._accumulate(mask * g / np.maximum(mask.sum(axis=axis, keepdims=True), 1))
        out._backward = _backward
        return out

    # --------------------------------------------------------------- reshaping

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        order = tuple(axes) if axes else None
        out = Tensor(self.data.transpose(order), self.requires_grad, (self,), "transpose")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if order is None:
                self._accumulate(out.grad.transpose())
            else:
                inverse = np.argsort(order)
                self._accumulate(out.grad.transpose(tuple(inverse)))
        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __getitem__(self, index) -> "Tensor":
        out = Tensor(self.data[index], self.requires_grad, (self,), "getitem")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g = np.zeros_like(self.data)
            np.add.at(g, index, out.grad)
            self._accumulate(g)
        out._backward = _backward
        return out

    def index_add(self, index: np.ndarray, n_out: int, axis: int = 0) -> "Tensor":
        """Scatter-add rows into `n_out` buckets: out[index[i]] += self[i].

        This is the aggregation primitive for message passing on graphs.
        """
        index = np.asarray(index, dtype=np.int64)
        shape = (n_out,) + self.shape[1:]
        values = np.zeros(shape, dtype=np.float64)
        np.add.at(values, index, self.data)
        out = Tensor(values, self.requires_grad, (self,), "index_add")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad[index])
        out._backward = _backward
        return out

    # ------------------------------------------------------------- activations

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def leaky_relu(self, slope: float = 0.01) -> "Tensor":
        out = Tensor(np.where(self.data > 0, self.data, slope * self.data),
                     self.requires_grad, (self,), "leaky_relu")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * np.where(self.data > 0, 1.0, slope))
        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        values = np.exp(np.clip(self.data, -700, 700))
        out = Tensor(values, self.requires_grad, (self,), "exp")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * values)
        out._backward = _backward
        return out

    def log(self, eps: float = 1e-12) -> "Tensor":
        out = Tensor(np.log(np.maximum(self.data, eps)), self.requires_grad, (self,), "log")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad / np.maximum(self.data, eps))
        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), self.requires_grad, (self,), "abs")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * np.sign(self.data))
        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        values = 1.0 / (1.0 + np.exp(-np.clip(self.data, -700, 700)))
        out = Tensor(values, self.requires_grad, (self,), "sigmoid")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * values * (1 - values))
        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        values = np.tanh(self.data)
        out = Tensor(values, self.requires_grad, (self,), "tanh")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (1 - values ** 2))
        out._backward = _backward
        return out

    def softplus(self) -> "Tensor":
        x = self.data
        values = np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -700, 30))))
        out = Tensor(values, self.requires_grad, (self,), "softplus")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad / (1.0 + np.exp(-np.clip(x, -700, 700))))
        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_sum = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        values = shifted - log_sum
        out = Tensor(values, self.requires_grad, (self,), "log_softmax")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            probs = np.exp(values)
            self._accumulate(out.grad - probs * out.grad.sum(axis=axis, keepdims=True))
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        return self.log_softmax(axis).exp()

    def clip(self, low: float, high: float) -> "Tensor":
        out = Tensor(np.clip(self.data, low, high), self.requires_grad, (self,), "clip")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                mask = (self.data >= low) & (self.data <= high)
                self._accumulate(out.grad * mask)
        out._backward = _backward
        return out


# --------------------------------------------------------------- free functions

def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape), requires_grad)


def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    tensors = list(tensors)
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis),
                 any(t.requires_grad for t in tensors), tuple(tensors), "concat")

    def _backward() -> None:
        if out.grad is None:
            return
        offset = 0
        for t in tensors:
            size = t.shape[axis]
            slicer = [slice(None)] * out.grad.ndim
            slicer[axis] = slice(offset, offset + size)
            if t.requires_grad:
                t._accumulate(out.grad[tuple(slicer)])
            offset += size
    out._backward = _backward
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return concat([t.reshape(t.shape[:axis] + (1,) + t.shape[axis:]) for t in tensors],
                  axis=axis)


def where(condition: np.ndarray, a: Tensor, b: Tensor) -> Tensor:
    condition = np.asarray(condition)
    out = Tensor(np.where(condition, a.data, b.data),
                 a.requires_grad or b.requires_grad, (a, b), "where")

    def _backward() -> None:
        if out.grad is None:
            return
        if a.requires_grad:
            a._accumulate(_unbroadcast(out.grad * condition, a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(out.grad * ~condition, b.shape))
    out._backward = _backward
    return out


def segment_mean(values: Tensor, index: np.ndarray, n_out: int) -> Tensor:
    """Mean of rows grouped by `index`; empty groups return zero."""
    index = np.asarray(index, dtype=np.int64)
    counts = np.zeros(n_out)
    np.add.at(counts, index, 1.0)
    denom = np.maximum(counts, 1.0).reshape((n_out,) + (1,) * (values.ndim - 1))
    return values.index_add(index, n_out) * (1.0 / denom)


def segment_max(values: Tensor, index: np.ndarray, n_out: int) -> Tensor:
    """Row-wise max grouped by `index`; empty groups return zero."""
    index = np.asarray(index, dtype=np.int64)
    shape = (n_out,) + values.shape[1:]
    out_data = np.full(shape, -np.inf)
    np.maximum.at(out_data, index, values.data)
    empty = ~np.isfinite(out_data)
    out_data[empty] = 0.0
    out = Tensor(out_data, values.requires_grad, (values,), "segment_max")

    def _backward() -> None:
        if out.grad is None or not values.requires_grad:
            return
        picked = (values.data == out_data[index])
        g = np.zeros_like(values.data)
        g[picked] = out.grad[index][picked]
        values._accumulate(g)
    out._backward = _backward
    return out


# ----------------------------------------------------------------- numerics

def finite_difference_grad(fn: Callable[[np.ndarray], float], x: np.ndarray,
                           eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient, used to verify analytic gradients."""
    grad = np.zeros_like(x, dtype=np.float64)
    flat = x.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = fn(x)
        flat[i] = original - eps
        minus = fn(x)
        flat[i] = original
        gflat[i] = (plus - minus) / (2 * eps)
    return grad
