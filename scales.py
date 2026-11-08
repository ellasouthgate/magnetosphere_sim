import numpy as np

import config as C


def signed_log(x, x0=C.SIGNED_LOG_X0_NT):
    """f(x) = sign(x) · ln(1 + |x|/x₀).  x and x₀ must be in the same units."""
    if x0 <= 0:
        raise ValueError("x0 must be positive")
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x) / x0)


def signed_log_inverse(y, x0=C.SIGNED_LOG_X0_NT):
    """Inverse of signed_log: x = sign(y) · x₀ · (exp|y| − 1)."""
    if x0 <= 0:
        raise ValueError("x0 must be positive")
    y = np.asarray(y, dtype=float)
    return np.sign(y) * x0 * np.expm1(np.abs(y))


def log10_magnitude(B, floor=C.LOG_B_FLOOR_NT):
    """log10|B|, floored so a null doesn't give log(0)."""
    mag = np.linalg.norm(np.asarray(B, dtype=float), axis=-1)
    return np.log10(np.maximum(mag, floor))


def scale_arrows(B, x0=C.SIGNED_LOG_X0_NT):
    """arrow = B̂ · ln(1 + |B|/x₀)"""
    B = np.asarray(B, dtype=float)
    mag = np.linalg.norm(B, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag > 0, B / mag, 0.0)
    return unit * signed_log(mag, x0)


def signed_log_ticks(y_max, x0=C.SIGNED_LOG_X0_NT):
    """Colour bar ticks for a signed log axis.  Returns (positions, labels)."""
    x_max = signed_log_inverse(y_max, x0)
    powers = 10.0 ** np.arange(np.floor(np.log10(x0)), np.log10(x_max) + 1e-9)
    powers = powers[powers <= x_max]
    values = np.concatenate([-powers[::-1], [0.0], powers])
    labels = [_label(v) for v in values]
    return signed_log(values, x0), labels


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _label(v):
    """0 → "0";  ±10ⁿ → "+10³" etc. (unicode, no LaTeX); otherwise "+2.5"."""
    if v == 0:
        return "0"
    sign = "+" if v > 0 else "−"
    n = np.log10(abs(v))
    if abs(n - round(n)) < 1e-9:
        n = int(round(n))
        if 0 <= n <= 2:
            return f"{sign}{10 ** n:g}"
        return f"{sign}10{str(n).translate(_SUPERSCRIPT)}"
    return f"{sign}{abs(v):g}"
