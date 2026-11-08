import numpy as np
import pytest

from scales import (signed_log, signed_log_inverse, log10_magnitude,
                    scale_arrows, signed_log_ticks)

X = np.concatenate([-np.logspace(-6, 6, 50), [0.0], np.logspace(-6, 6, 50)])


@pytest.mark.parametrize("x0", [1e-3, 1.0, 31.0])
def test_f_of_zero_is_zero(x0):
    assert signed_log(0.0, x0) == 0.0


@pytest.mark.parametrize("x0", [1e-3, 1.0, 31.0])
def test_f_is_odd(x0):
    assert np.array_equal(signed_log(-X, x0), -signed_log(X, x0))


@pytest.mark.parametrize("x0", [1e-3, 1.0, 31.0])
def test_slope_at_zero_is_one_over_x0(x0):
    for h in (1e-6 * x0, 1e-9 * x0):
        slope = (signed_log(h, x0) - signed_log(-h, x0)) / (2 * h)
        assert slope == pytest.approx(1 / x0, rel=1e-6)


def test_continuous_at_zero():
    eps = np.logspace(-15, -8, 8)
    assert np.all(np.abs(signed_log(eps)) < 1e-7)
    assert np.all(np.abs(signed_log(-eps)) < 1e-7)
    assert np.allclose(signed_log(eps), -signed_log(-eps))


def test_matches_the_two_branch_definition():
    x0 = 2.5
    pos, neg = X[X >= 0], X[X < 0]
    assert np.allclose(signed_log(pos, x0), np.log(1 + pos / x0))
    assert np.allclose(signed_log(neg, x0), -np.log(1 - neg / x0))


def test_linear_near_zero_logarithmic_far_away():
    x0 = 1.0
    assert signed_log(1e-3, x0) == pytest.approx(1e-3, rel=1e-3)
    assert signed_log(1e6, x0) - signed_log(1e5, x0) == pytest.approx(np.log(10), rel=1e-4)


def test_monotonic():
    assert np.all(np.diff(signed_log(np.sort(X))) > 0)


def test_inverse():
    for x0 in (0.1, 1.0, 50.0):
        assert np.allclose(signed_log_inverse(signed_log(X, x0), x0), X, rtol=1e-10,
                           atol=1e-15)


def test_bad_x0_raises():
    with pytest.raises(ValueError):
        signed_log(1.0, 0.0)
    with pytest.raises(ValueError):
        signed_log_inverse(1.0, -1.0)


def test_log10_magnitude():
    B = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    out = log10_magnitude(B, floor=1e-3)
    assert np.allclose(out, [np.log10(5.0), -3.0, 3.0])


def test_scale_arrows_keeps_direction_and_scales_length():
    B = np.array([[3.0, 0.0, 4.0], [0.0, -2000.0, 0.0], [0.0, 0.0, 0.0]])
    A = scale_arrows(B, x0=1.0)
    assert np.allclose(A[0] / np.linalg.norm(A[0]), [0.6, 0.0, 0.8])
    assert np.allclose(A[1] / np.linalg.norm(A[1]), [0.0, -1.0, 0.0])
    assert np.linalg.norm(A[0]) == pytest.approx(np.log(1 + 5.0))
    assert np.linalg.norm(A[1]) == pytest.approx(np.log(1 + 2000.0))
    assert np.array_equal(A[2], [0.0, 0.0, 0.0])


def test_signed_log_ticks():
    x0 = 1.0
    pos, labels = signed_log_ticks(signed_log(5e4, x0), x0)
    assert labels[len(labels) // 2] == "0"
    assert np.allclose(pos, -pos[::-1])                       # symmetric
    assert labels == ["−10⁴", "−10³", "−100", "−10", "−1", "0",
                      "+1", "+10", "+100", "+10³", "+10⁴"]
    assert np.all(np.abs(pos) <= signed_log(5e4, x0) + 1e-12)
