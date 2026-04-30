"""R-peak detection.

This is a deliberately small, self-contained Pan-Tompkins-flavoured
detector. It is *not* meant to compete with research-grade libraries
(e.g. neurokit2); it is meant to give plausible R-peak locations on
clean, pre-bandpassed ECG with no extra dependencies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal


def _moving_window_integrate(x: np.ndarray, win: int) -> np.ndarray:
    """Centred uniform moving average."""
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float,
    *,
    min_distance_s: float = 0.30,
    min_height_z: float = 0.6,
) -> np.ndarray:
    """Return integer R-peak indices.

    Parameters
    ----------
    ecg:
        1-D ECG, ideally already bandpass-filtered ~5-25 Hz. The function
        will additionally apply a derivative + squaring step so passing
        broadband ECG works too, just less reliably.
    fs:
        Sampling frequency in Hz.
    min_distance_s:
        Refractory minimum between successive R peaks, in seconds.
        300 ms accommodates rates up to 200 bpm.
    min_height_z:
        Minimum peak height as a fraction of the *MWI* signal's robust
        scale (median absolute deviation). 0.6 works on clean ECG; the
        caller can drop it for noisy records.

    Notes
    -----
    Returns an empty array on degenerate input rather than raising; this
    lets the synthesizer fall back to a sinusoidal RESP / regularised PPG
    without a try/except dance at every call site.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    if ecg.size < int(fs):
        return np.empty(0, dtype=np.int64)

    # Pan-Tompkins: derivative -> square -> moving-window integrate.
    deriv = np.diff(ecg, prepend=ecg[0])
    sq = deriv * deriv
    win = max(1, int(round(0.150 * fs)))   # 150 ms integration window
    mwi = _moving_window_integrate(sq, win)

    # Percentile-based threshold + prominence: separates QRS from T-waves.
    # Pure MAD on a sparse-spike signal collapses near 0, so use the
    # 50th/95th-percentile gap to set the height floor.
    finite = mwi[np.isfinite(mwi)]
    if finite.size == 0 or np.allclose(finite, finite[0]):
        return np.empty(0, dtype=np.int64)
    p50 = float(np.percentile(finite, 50))
    p95 = float(np.percentile(finite, 95))
    span = max(p95 - p50, 1e-9)
    height = p50 + min_height_z * span
    prominence = 0.4 * span

    distance = max(1, int(round(min_distance_s * fs)))
    peaks, _ = sp_signal.find_peaks(
        mwi, height=height, distance=distance, prominence=prominence,
    )

    # Refine each peak by snapping to the local maximum on the raw ECG
    # (within +-50 ms). MWI peaks are slightly shifted relative to the QRS.
    half = max(1, int(round(0.05 * fs)))
    refined = []
    for p in peaks:
        lo, hi = max(0, p - half), min(ecg.size, p + half + 1)
        if hi <= lo:
            continue
        refined.append(lo + int(np.argmax(np.abs(ecg[lo:hi]))))
    return np.asarray(refined, dtype=np.int64)


def rr_intervals_seconds(peak_indices: np.ndarray, fs: float) -> np.ndarray:
    """Return RR intervals in seconds; empty array when fewer than 2 peaks."""
    if peak_indices.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(peak_indices.astype(np.float64)) / float(fs)


def heart_rate_bpm(peak_indices: np.ndarray, fs: float) -> Tuple[float, float]:
    """Return (median_bpm, std_bpm) over the given peak set.

    Falls back to ``(60.0, 0.0)`` when fewer than 2 peaks are present so
    that downstream vitals always have a numeric value.
    """
    rr = rr_intervals_seconds(peak_indices, fs)
    if rr.size == 0:
        return 60.0, 0.0
    bpm = 60.0 / rr
    return float(np.median(bpm)), float(np.std(bpm))
