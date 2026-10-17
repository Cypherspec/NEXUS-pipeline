"""Every autograd operation checked against central finite differences."""

import numpy as np
import pytest

from nexus.models.autograd import (
    Tensor,
    concat,
    finite_difference_grad,
    segment_max,
    segment_mean,
    stack,
    where,
    zeros,
)

RNG = np.random.default_rng(0)


def check_grad(fn, *shapes, tol=1e-5, positive=False):
    """Compare analytic gradients of `fn` against finite differences."""
    arrays = []
    for shape in shapes:
        a = RNG.normal(size=shape)
        if positive:
            a = np.abs(a) + 0.5
        arrays.append(a)
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    assert out.data.size == 1, "test functions must return a scalar"
    out.backward()
    for k, (t, a) in enumerate(zip(tensors, arrays)):
        def scalar(_x, k=k):
            ts = [Tensor(arr.copy()) for arr in arrays]
            return float(fn(*ts).data)
        numeric = finite_difference_grad(scalar, arrays[k])
        assert t.grad is not None, f"input {k} received no gradient"
        np.testing.assert_allclose(t.grad, numeric, atol=tol, rtol=tol,
                                   err_msg=f"gradient mismatch for input {k}")


# ------------------------------------------------------------------ arithmetic

def test_add_grad():
    check_grad(lambda a, b: (a + b).sum(), (3, 4), (3, 4))


def test_add_broadcast_grad():
    check_grad(lambda a, b: (a + b).sum(), (3, 4), (4,))
    check_grad(lambda a, b: (a + b).sum(), (3, 4), (1, 4))


def test_mul_grad():
    check_grad(lambda a, b: (a * b).sum(), (3, 4), (3, 4))


def test_mul_broadcast_grad():
    check_grad(lambda a, b: (a * b).sum(), (2, 3, 4), (4,))


def test_sub_and_neg_grad():
    check_grad(lambda a, b: (a - b).sum(), (3, 4), (3, 4))
    check_grad(lambda a: (-a).sum(), (3, 4))


def test_div_grad():
    check_grad(lambda a, b: (a / b).sum(), (3, 4), (3, 4), positive=True)


def test_pow_grad():
    check_grad(lambda a: (a ** 3).sum(), (3, 4))
    check_grad(lambda a: (a ** 0.5).sum(), (3, 4), positive=True)


def test_matmul_grad():
    check_grad(lambda a, b: (a @ b).sum(), (3, 4), (4, 2))


def test_batched_matmul_grad():
    check_grad(lambda a, b: (a @ b).sum(), (2, 3, 4), (2, 4, 5))


def test_rsub_rtruediv():
    a = Tensor(np.array([2.0]), requires_grad=True)
    out = (1.0 - a) + (6.0 / a)
    out.backward()
    assert a.grad == pytest.approx(-1.0 - 6.0 / 4.0)


# ------------------------------------------------------------------ reductions

def test_sum_axis_grad():
    check_grad(lambda a: a.sum(axis=0).sum(), (3, 4))
    check_grad(lambda a: a.sum(axis=1, keepdims=True).sum(), (3, 4))


def test_mean_grad():
    check_grad(lambda a: a.mean().sum(), (3, 4))
    check_grad(lambda a: a.mean(axis=1).sum(), (3, 4))


def test_mean_value_matches_numpy():
    x = np.arange(12.0).reshape(3, 4)
    assert Tensor(x).mean().data == pytest.approx(x.mean())
    np.testing.assert_allclose(Tensor(x).mean(axis=1).data, x.mean(axis=1))


def test_max_grad():
    check_grad(lambda a: a.max().sum(), (3, 4))
    check_grad(lambda a: a.max(axis=1).sum(), (3, 4))


# ------------------------------------------------------------------- reshaping

def test_reshape_grad():
    check_grad(lambda a: (a.reshape(4, 3) * 2.0).sum(), (3, 4))


def test_transpose_grad():
    check_grad(lambda a: (a.T * 3.0).sum(), (3, 4))
    check_grad(lambda a: (a.transpose(1, 0, 2)).sum(), (2, 3, 4))


def test_getitem_grad():
    check_grad(lambda a: a[1:3].sum(), (5, 4))
    check_grad(lambda a: a[:, 0].sum(), (5, 4))


def test_getitem_with_repeated_index_accumulates():
    x = Tensor(np.zeros((3, 2)), requires_grad=True)
    out = x[np.array([0, 0, 2])].sum()
    out.backward()
    np.testing.assert_allclose(x.grad, [[2, 2], [0, 0], [1, 1]])


def test_concat_and_stack_grad():
    check_grad(lambda a, b: concat([a, b], axis=0).sum(), (2, 3), (4, 3))
    check_grad(lambda a, b: concat([a, b], axis=1).sum(), (2, 3), (2, 5))
    check_grad(lambda a, b: stack([a, b]).sum(), (2, 3), (2, 3))


def test_where_grad():
    cond = RNG.normal(size=(3, 4)) > 0
    check_grad(lambda a, b: where(cond, a, b).sum(), (3, 4), (3, 4))


# ----------------------------------------------------------------- activations

@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "exp", "softplus", "abs"])
def test_activation_grads(name):
    # ReLU and abs are non-differentiable at 0; random inputs avoid it.
    check_grad(lambda a: getattr(a, name)().sum(), (4, 5))


def test_leaky_relu_grad():
    check_grad(lambda a: a.leaky_relu(0.1).sum(), (4, 5))


def test_log_and_sqrt_grad():
    check_grad(lambda a: a.log().sum(), (4, 5), positive=True)
    check_grad(lambda a: a.sqrt().sum(), (4, 5), positive=True)


def test_log_softmax_grad():
    check_grad(lambda a: (a.log_softmax(axis=-1) * 1.5).sum(), (4, 5))


def test_softmax_is_normalised_and_differentiable():
    x = Tensor(RNG.normal(size=(4, 5)), requires_grad=True)
    p = x.softmax(axis=-1)
    np.testing.assert_allclose(p.data.sum(axis=-1), 1.0, atol=1e-12)
    check_grad(lambda a: (a.softmax(axis=-1) ** 2).sum(), (4, 5))


def test_clip_grad():
    check_grad(lambda a: a.clip(-0.5, 0.5).sum(), (4, 5))


# -------------------------------------------------------------- graph scatter

def test_index_add_grad():
    idx = np.array([0, 0, 1, 2, 2])
    check_grad(lambda a: (a.index_add(idx, 3) ** 2).sum(), (5, 3))


def test_index_add_values():
    x = Tensor(np.array([[1.0], [2.0], [3.0]]))
    out = x.index_add(np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(out.data, [[3.0], [3.0]])


def test_segment_mean_grad_and_values():
    idx = np.array([0, 0, 1, 2, 2])
    check_grad(lambda a: (segment_mean(a, idx, 3) ** 2).sum(), (5, 3))
    x = Tensor(np.array([[1.0], [3.0], [10.0]]))
    np.testing.assert_allclose(segment_mean(x, np.array([0, 0, 1]), 2).data,
                               [[2.0], [10.0]])


def test_segment_mean_empty_group_is_zero():
    x = Tensor(np.array([[5.0]]))
    np.testing.assert_allclose(segment_mean(x, np.array([0]), 3).data,
                               [[5.0], [0.0], [0.0]])


def test_segment_max_grad_and_values():
    idx = np.array([0, 0, 1, 2, 2])
    check_grad(lambda a: (segment_max(a, idx, 3) ** 2).sum(), (5, 3))
    x = Tensor(np.array([[1.0], [3.0], [10.0]]))
    np.testing.assert_allclose(segment_max(x, np.array([0, 0, 1]), 2).data,
                               [[3.0], [10.0]])


# ------------------------------------------------------------------ mechanics

def test_gradient_accumulates_across_uses():
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = x * x + x          # dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_deep_chain_matches_finite_difference():
    def fn(a):
        h = a
        for _ in range(6):
            h = (h @ h.T).tanh() @ h
        return h.sum()
    check_grad(fn, (3, 3), tol=1e-4)


def test_no_grad_for_constant_inputs():
    x = Tensor(np.ones((2, 2)))                  # requires_grad=False
    w = Tensor(np.ones((2, 2)), requires_grad=True)
    (x @ w).sum().backward()
    assert x.grad is None and w.grad is not None


def test_backward_on_non_scalar_requires_gradient():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
    (x * 2).backward(np.ones((2, 2)))
    np.testing.assert_allclose(x.grad, 2.0)


def test_zero_grad_and_detach():
    x = Tensor(np.ones(3), requires_grad=True)
    (x * 2).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None
    d = x.detach()
    assert not d.requires_grad and d.data is not x.data


def test_zeros_helper():
    z = zeros(2, 3, requires_grad=True)
    assert z.shape == (2, 3) and z.requires_grad and not z.data.any()
