"""Rational/non-rational resampling with anti-aliasing.

We use ``scipy.signal.resample_poly`` everywhere because it is anti-aliased,
fast, and behaves well for both rational ratios (e.g. 360 -> 200) and
arbitrary ratios (e.g. 257 -> 200) once we approximate them with integer
up/down factors.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import numpy as np
from scipy import signal as sp_signal


_MAX_RATIO_DENOM = 1_000  # cap polyphase work; sufficient resolution for fs


def _rational_factors(fs_in: float, fs_out: float) -> Tuple[int, int]:
    """Approximate fs_out / fs_in as a small fraction up/down.

    ``Fraction.limit_denominator`` gives the closest rational under the cap.
    For real-world ECG sampling rates the approximation error is well
    below 1e-4 Hz, which is negligible for arrhythmia analysis.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"sampling rates must be positive, got {fs_in}, {fs_out}")
    frac = Fraction(fs_out / fs_in).limit_denominator(_MAX_RATIO_DENOM)
    up, down = frac.numerator, frac.denominator
    if up <= 0 or down <= 0:
        raise ValueError(f"degenerate resample ratio {up}/{down}")
    return up, down


class Resampler:
    """Wrapper around ``resample_poly`` with deterministic output length."""

    @staticmethod
    def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
        """Resample a 1-D signal from ``fs_in`` to ``fs_out``.

        The returned length is approximately ``len(x) * fs_out / fs_in``;
        callers that require an exact length should slice/pad afterwards
        (see :func:`resample_to_length`).
        """
        if x.ndim != 1:
            raise ValueError(f"expected 1-D signal, got shape {x.shape}")
        if abs(fs_in - fs_out) < 1e-9:
            return x.astype(np.float32, copy=True)
        up, down = _rational_factors(fs_in, fs_out)
        y = sp_signal.resample_poly(x.astype(np.float64), up, down)
        return y.astype(np.float32)

    @staticmethod
    def resample_to_length(
        x: np.ndarray, fs_in: float, fs_out: float, n_out: int
    ) -> np.ndarray:
        """Resample then crop/zero-pad to exactly ``n_out`` samples.

        Polyphase resampling can leave a 1-sample drift; crop/pad keeps the
        downstream HDF5 shape contract exact.
        """
        y = Resampler.resample(x, fs_in, fs_out)
        if y.size == n_out:
            return y
        if y.size > n_out:
            return y[:n_out].copy()
        out = np.zeros(n_out, dtype=np.float32)
        out[: y.size] = y
        return out
