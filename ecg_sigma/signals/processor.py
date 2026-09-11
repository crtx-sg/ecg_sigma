"""ECG pre-processing: bandpass, notch, NaN handling and SNR estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


@dataclass(frozen=True)
class ProcessorConfig:
    """Pre-processor knobs. Defaults match the YAML defaults."""

    bandpass_hz: Tuple[float, float] = (0.5, 40.0)
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0


class SignalProcessor:
    """Stateless utility methods. Instantiated per-pipeline so that we can
    bind a single :class:`ProcessorConfig` for every record processed in a
    run."""

    def __init__(self, cfg: ProcessorConfig = ProcessorConfig()):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # NaN / inf hygiene
    # ------------------------------------------------------------------ #
    @staticmethod
    def sanitize(x: np.ndarray) -> np.ndarray:
        """Linear-interpolate NaNs and clip Infs in place-safe fashion.

        WFDB occasionally yields NaNs at record boundaries or where the
        original recording had a dropout; leaving them in would propagate
        through the filter response and corrupt the entire window.
        """
        y = np.asarray(x, dtype=np.float64).copy()
        finite = np.isfinite(y)
        if finite.all():
            return y
        if not finite.any():
            return np.zeros_like(y)
        idx = np.arange(y.size)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
        return y

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def bandpass(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Zero-phase 4th-order Butterworth bandpass."""
        lo, hi = self.cfg.bandpass_hz
        nyq = 0.5 * fs
        if lo <= 0 or hi >= nyq:
            raise ValueError(
                f"bandpass {lo}-{hi} Hz invalid for fs={fs} (nyq={nyq})"
            )
        sos = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        return sp_signal.sosfiltfilt(sos, x).astype(np.float64)

    def notch(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Optional power-line notch."""
        if not self.cfg.notch_hz:
            return x
        if self.cfg.notch_hz >= 0.5 * fs:
            return x
        b, a = sp_signal.iirnotch(self.cfg.notch_hz, self.cfg.notch_q, fs)
        return sp_signal.filtfilt(b, a, x).astype(np.float64)

    def preprocess(self, x: np.ndarray, fs: float) -> np.ndarray:
        """End-to-end clean: sanitize -> bandpass -> notch."""
        y = self.sanitize(x)
        y = self.bandpass(y, fs)
        y = self.notch(y, fs)
        return y.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Quality scoring
    # ------------------------------------------------------------------ #
    # Bands used by :meth:`quality_score`. All three sit *inside* the
    # pre-processor's passband, so the score measures the signal rather
    # than the filter's stopband.
    QRS_BAND_HZ = (5.0, 15.0)
    BASELINE_BAND_HZ = (0.5, 5.0)     # drift, motion, respiration artefact
    HF_NOISE_BAND_HZ = (15.0, 40.0)   # EMG / muscle / electrode noise

    @classmethod
    def quality_score(cls, x: np.ndarray, fs: float) -> float:
        """Heuristic [0, 1] in-band SQI for a pre-processed ECG window.

        ``quality = (1 - nan_ratio) * qrs / (qrs + baseline + hf_noise)``

        where each term is mean Welch band power. All three bands lie
        inside the 0.5-40 Hz passband that :meth:`preprocess` leaves
        behind: an earlier version measured "noise" above 40 Hz, which is
        the bandpass *stopband*, so every window scored ~1.0 regardless of
        content. A flat window scores 0.

        Still deliberately crude -- it separates "unusable" from "mostly
        OK" for downstream QC. It is not a clinical-grade metric.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0
        nan_ratio = float(np.mean(~np.isfinite(x)))
        x_clean = x[np.isfinite(x)]
        if x_clean.size < int(fs):  # < 1 s of usable data
            return max(0.0, 1.0 - nan_ratio) * 0.1
        if float(np.ptp(x_clean)) <= 0.0:
            return 0.0              # flat channel carries no signal

        # Welch PSD; reasonable nperseg for short windows.
        nperseg = min(x_clean.size, int(2 * fs))
        f, pxx = sp_signal.welch(x_clean, fs=fs, nperseg=nperseg)

        def band_power(lo: float, hi: float) -> float:
            mask = (f >= lo) & (f < hi)
            return float(np.mean(pxx[mask])) if mask.any() else 0.0

        qrs = band_power(*cls.QRS_BAND_HZ)
        baseline = band_power(*cls.BASELINE_BAND_HZ)
        hf = band_power(*cls.HF_NOISE_BAND_HZ)
        total = qrs + baseline + hf
        if total <= 0.0:
            return 0.0
        return float(np.clip((1.0 - nan_ratio) * (qrs / total), 0.0, 1.0))
