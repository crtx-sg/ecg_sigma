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
    @staticmethod
    def quality_score(x: np.ndarray, fs: float) -> float:
        """Heuristic [0, 1] quality score combining NaN ratio and SNR.

        - NaN ratio -> direct linear penalty.
        - SNR proxy -> ratio of band-power in 5-15 Hz (QRS band) to power
          in 40-(fs/2) Hz (likely noise). Mapped through a saturating
          function so saturated values do not dominate the score.

        This is intentionally crude: it discriminates "totally bad" from
        "mostly OK" segments and is good enough to flag for downstream
        QC. It is not a clinical-grade quality metric.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0
        nan_ratio = float(np.mean(~np.isfinite(x)))
        x_clean = x[np.isfinite(x)]
        if x_clean.size < int(fs):  # < 1 s of usable data
            return max(0.0, 1.0 - nan_ratio) * 0.1

        # Welch PSD; reasonable nperseg for short windows.
        nperseg = min(x_clean.size, int(2 * fs))
        f, pxx = sp_signal.welch(x_clean, fs=fs, nperseg=nperseg)
        qrs_band = (f >= 5.0) & (f <= 15.0)
        noise_band = f >= 40.0
        qrs_power = float(np.mean(pxx[qrs_band])) if qrs_band.any() else 0.0
        noise_power = float(np.mean(pxx[noise_band])) if noise_band.any() else 1e-12
        snr = qrs_power / max(noise_power, 1e-12)
        # squashed: 0 at snr=0, ~0.95 at snr=20
        snr_score = snr / (snr + 1.0)
        return float(np.clip((1.0 - nan_ratio) * snr_score, 0.0, 1.0))
