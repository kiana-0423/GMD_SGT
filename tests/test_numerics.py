"""Numerical robustness of the cutoff envelope and envelope-weighted attention."""

from __future__ import annotations

import math

import pytest
import torch

from gmd_sgt.models.radial import PolynomialCutoff
from gmd_sgt.models.transformer_correction import _segment_softmax

CUTOFF = 3.0


def _expanded_envelope(r: torch.Tensor, p: int) -> torch.Tensor:
    """The textbook (expanded) polynomial, evaluated in float64 as reference."""
    u = r.double() / CUTOFF
    env = (
        1.0
        - (p + 1) * (p + 2) / 2 * u**p
        + p * (p + 2) * u ** (p + 1)
        - p * (p + 1) / 2 * u ** (p + 2)
    )
    return env * (u < 1.0)


# ── Envelope ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("p", [3, 6, 9])
def test_envelope_equals_reference_polynomial(p):
    r = torch.linspace(0.0, 3.3, 20001, dtype=torch.float64)
    torch.testing.assert_close(PolynomialCutoff(CUTOFF, p)(r), _expanded_envelope(r, p), atol=1e-13, rtol=0)


def test_float32_envelope_is_nonnegative_and_accurate_near_cutoff():
    env = PolynomialCutoff(CUTOFF)
    r32 = torch.linspace(2.9, 3.0, 400001, dtype=torch.float32)
    r32 = torch.cat([r32, torch.tensor([2.99365497], dtype=torch.float32)])
    e32 = env(r32)
    e64 = env(r32.double())  # same input values, float64 arithmetic
    assert (e32 >= 0).all(), float(e32.min())
    inside = e64 > 0
    rel = ((e32.double() - e64).abs() / e64)[inside]
    assert rel.max() < 1e-5, float(rel.max())
    # The case that used to be -5.7e-6 in float32.
    value = float(env(torch.tensor([2.99365497], dtype=torch.float32)))
    assert value == pytest.approx(5.25629e-7, rel=1e-4)


def test_envelope_value_and_two_derivatives_vanish_at_cutoff():
    env = PolynomialCutoff(CUTOFF)
    for delta in (1e-2, 1e-3, 1e-4):
        r = torch.tensor([CUTOFF - delta], dtype=torch.float64, requires_grad=True)
        f = env(r)
        (d1,) = torch.autograd.grad(f.sum(), r, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), r)
        # f ~ delta^3, f' ~ delta^2, f'' ~ delta near the cutoff
        assert 0 < float(f) < 30 * delta**3
        assert abs(float(d1)) < 100 * delta**2
        assert abs(float(d2)) < 200 * delta
    r = torch.tensor([CUTOFF, CUTOFF + 0.1], dtype=torch.float64, requires_grad=True)
    f = env(r)
    (d1,) = torch.autograd.grad(f.sum(), r)
    assert torch.equal(f.detach(), torch.zeros(2, dtype=torch.float64))
    assert torch.equal(d1, torch.zeros(2, dtype=torch.float64))


# ── Envelope-weighted attention ─────────────────────────────────────────────


def _reference_attention(scores, index, dim_size, weight):
    """Direct float64 evaluation of w exp(s) / (1 + sum w exp(s))."""
    scores, weight = scores.double(), weight.double()
    out = torch.zeros_like(scores)
    for node in range(dim_size):
        mask = index == node
        terms = weight[mask, None] * scores[mask].exp()
        out[mask] = terms / (1.0 + terms.sum(0))
    return out


def test_zero_weight_edge_is_equivalent_to_removing_it():
    scores = torch.tensor([[40.0], [0.0]])
    index = torch.tensor([0, 0])
    with_zero = _segment_softmax(scores, index, 1, torch.tensor([0.0, 1.0]))
    removed = _segment_softmax(scores[1:], index[1:], 1, torch.tensor([1.0]))
    assert with_zero.flatten().tolist() == [0.0, 0.5]
    torch.testing.assert_close(with_zero[1:], removed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighted_attention_matches_formula_for_extreme_inputs(dtype):
    scores = torch.tensor(
        [[60.0, -3.0], [2.0, 80.0], [0.0, 1.0], [-5.0, 30.0], [20.0, 20.0], [85.0, -85.0]],
        dtype=dtype,
    )
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    weight = torch.tensor([1e-20, 0.3, 1e-35, 0.0, 5.256e-7, 1e-30], dtype=dtype)
    out = _segment_softmax(scores, index, 4, weight)  # node 3 has no edges
    assert torch.isfinite(out).all() and (out >= 0).all()
    torch.testing.assert_close(out.double(), _reference_attention(scores, index, 4, weight), atol=1e-6, rtol=1e-5)
    # Probabilities over the edges plus the null key never exceed one.
    per_node = torch.zeros(4, 2, dtype=torch.float64).index_add(0, index, out.double())
    assert (per_node <= 1.0 + 1e-6).all()


def test_weighted_attention_handles_empty_edges():
    out = _segment_softmax(
        torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 5, torch.zeros(0)
    )
    assert out.shape == (0, 3)


def test_weighted_attention_has_finite_first_and_second_derivatives():
    scores = torch.tensor([[40.0], [0.0], [25.0], [3.0]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.0, 1.0, 1e-25, 0.4], dtype=torch.float64, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1])
    out = _segment_softmax(scores, index, 2, weight)
    loss = (out * torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)).sum()
    g_scores, g_weight = torch.autograd.grad(loss, (scores, weight), create_graph=True)
    assert torch.isfinite(g_scores).all() and torch.isfinite(g_weight).all()
    (gg_scores,) = torch.autograd.grad((g_scores**2).sum() + g_weight[1:].sum(), scores)
    assert torch.isfinite(gg_scores).all()
    assert g_scores[0].item() == 0.0  # zero-weight edge has no influence


def test_weighted_attention_gradients_match_finite_differences():
    torch.manual_seed(0)
    scores = (torch.randn(5, 2, dtype=torch.float64) * 4).requires_grad_(True)
    weight = torch.tensor([0.7, 1e-3, 0.2, 0.9, 0.05], dtype=torch.float64, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 1])
    assert torch.autograd.gradcheck(lambda s, w: _segment_softmax(s, index, 2, w), (scores, weight))
    assert torch.autograd.gradgradcheck(lambda s, w: _segment_softmax(s, index, 2, w), (scores, weight))


def test_negative_free_envelope_feeds_bounded_attention_in_float32():
    """The audit case: envelope at r=2.99365497 with logits up to 20."""
    w = PolynomialCutoff(CUTOFF)(torch.tensor([2.99365497], dtype=torch.float32))
    for logit in (0.0, 10.0, 20.0):
        a = float(_segment_softmax(torch.tensor([[logit]]), torch.tensor([0]), 1, w))
        expected = 5.25629e-7 * math.exp(logit) / (1 + 5.25629e-7 * math.exp(logit))
        assert 0.0 <= a <= 1.0
        assert a == pytest.approx(expected, rel=1e-3)
