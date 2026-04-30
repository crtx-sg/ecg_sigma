"""PPG and respiration synthesis.

Both modalities are absent in MIT-BIH/INCART/PTB-XL. We *derive* them from
the ECG (PPG from R-peaks, RESP from R-peak amplitude modulation) when
possible, else fall back to a rule-based simulation. Each waveform is
returned at its target sampling rate and length so the writer can drop it
straight into the HDF5 layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from ..schema import (
    METHOD_DERIVED_FROM_ECG,
    METHOD_RULE_BASED,
    SOURCE_SYNTHETIC,
    SchemaSpec,
    ExtrasTag,
)
from .peaks import detect_r_peaks


@dataclass(frozen=True)
class SynthesisConfig:
    """Tuning knobs for PPG/RESP synthesis."""

    ppg_pulse_delay_s: float = 0.20      # ECG R-peak -> PPG foot
    ppg_systolic_width_s: float = 0.10
    ppg_dicrotic_offset_s: float = 0.30
    ppg_dicrotic_amp: float = 0.35
    resp_method: str = "edr"             # 'edr' or 'sinusoidal'
    resp_default_rate_brpm: float = 15.0
    resp_noise_std: float = 0.05


@dataclass
class SynthesisResult:
    """Container holding the two synthesised waveforms + extras tags."""

    ppg: np.ndarray
    ppg_extras: ExtrasTag
    resp: np.ndarray
    resp_extras: ExtrasTag


class ModalitiesSynthesizer:
    """Generate PPG and RESP from a 12-second ECG window.

    Both methods accept the *post-resampled* ECG window already at
    ``schema.ecg_fs``. R-peak detection is run on Lead II (``ECG2``) which
    is the most universally available limb lead; callers should ensure the
    array indexing uses the canonical lead order.
    """

    def __init__(self, schema: SchemaSpec, cfg: SynthesisConfig, rng: np.random.Generator):
        self.schema = schema
        self.cfg = cfg
        self.rng = rng

    # ------------------------------------------------------------------ #
    # Top-level
    # ------------------------------------------------------------------ #
    def synthesize(self, ecg_lead_ii: np.ndarray) -> SynthesisResult:
        """Generate PPG (75 Hz) and RESP (33.33 Hz) for the input window."""
        ppg, ppg_method = self._ppg_from_ecg(ecg_lead_ii)
        resp, resp_method, resp_notes = self._resp_from_ecg(ecg_lead_ii)
        return SynthesisResult(
            ppg=ppg,
            ppg_extras=ExtrasTag(
                source=SOURCE_SYNTHETIC,
                method=ppg_method,
                notes=f"delay_s={self.cfg.ppg_pulse_delay_s}",
            ),
            resp=resp,
            resp_extras=ExtrasTag(
                source=SOURCE_SYNTHETIC,
                method=resp_method,
                notes=resp_notes,
            ),
        )

    # ------------------------------------------------------------------ #
    # PPG
    # ------------------------------------------------------------------ #
    def _ppg_from_ecg(self, ecg: np.ndarray) -> tuple[np.ndarray, str]:
        """Generate a PPG waveform shaped by the R-peak times."""
        n_out = self.schema.ppg_samples
        fs_out = self.schema.ppg_fs
        peaks = detect_r_peaks(ecg, self.schema.ecg_fs)

        if peaks.size < 2:
            # Fall back to a regular pulse train at a default 75 bpm.
            return self._ppg_regular(75.0, n_out, fs_out), METHOD_RULE_BASED

        # Convert R-peak indices (in ECG fs) to PPG samples and offset.
        ppg_t = (peaks / self.schema.ecg_fs) + self.cfg.ppg_pulse_delay_s
        ppg = np.zeros(n_out, dtype=np.float64)
        for i, t in enumerate(ppg_t):
            # Use the previous beat to size this pulse so dicrotic notch
            # placement adapts to the current heart rate.
            if i == 0:
                rr = (ppg_t[1] - ppg_t[0]) if ppg_t.size > 1 else 0.8
            else:
                rr = ppg_t[i] - ppg_t[i - 1]
            self._add_pulse(ppg, t, fs_out, rr)
        # Mild low-pass to remove discontinuities.
        ppg = self._lowpass(ppg, fs_out, 8.0)
        # Normalise into a reasonable amplitude range (peak-to-peak ~1.0).
        amp = np.ptp(ppg)
        if amp > 0:
            ppg = ppg / amp
        return ppg.astype(np.float32), METHOD_DERIVED_FROM_ECG

    def _add_pulse(
        self, buf: np.ndarray, t_start: float, fs: float, rr: float
    ) -> None:
        """Place a 2-Gaussian (systolic + dicrotic) pulse into ``buf``."""
        n = buf.size
        idx_start = int(round(t_start * fs))
        if idx_start >= n or idx_start < -int(0.5 * fs):
            return

        # Pulse half-life proportional to RR. Longer RR -> wider pulse.
        sys_w = max(self.cfg.ppg_systolic_width_s, 0.07 * rr)
        dic_off = self.cfg.ppg_dicrotic_offset_s + 0.05 * (rr - 0.8)

        # Time vector spanning ~0.7 s of pulse, clipped to buffer.
        span = max(int(round(0.7 * fs)), int(round((dic_off + 0.2) * fs)))
        i0 = max(0, idx_start)
        i1 = min(n, idx_start + span)
        if i1 <= i0:
            return
        t = (np.arange(i0, i1) - idx_start) / fs

        sigma_sys = sys_w / 2.355  # FWHM -> sigma
        sigma_dic = sigma_sys * 1.6
        systolic = np.exp(-0.5 * ((t - sys_w) / sigma_sys) ** 2)
        dicrotic = self.cfg.ppg_dicrotic_amp * np.exp(
            -0.5 * ((t - dic_off) / sigma_dic) ** 2
        )
        buf[i0:i1] += systolic + dicrotic

    def _ppg_regular(self, hr_bpm: float, n_out: int, fs_out: float) -> np.ndarray:
        """Generate a metronome-paced PPG when R-peaks cannot be found."""
        buf = np.zeros(n_out, dtype=np.float64)
        rr = 60.0 / max(hr_bpm, 1.0)
        t_start = 0.2  # initial offset
        while t_start < n_out / fs_out:
            self._add_pulse(buf, t_start, fs_out, rr)
            t_start += rr
        buf = self._lowpass(buf, fs_out, 8.0)
        amp = np.ptp(buf)
        if amp > 0:
            buf = buf / amp
        return buf.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Respiration
    # ------------------------------------------------------------------ #
    def _resp_from_ecg(self, ecg: np.ndarray) -> tuple[np.ndarray, str, str]:
        """Generate a respiration waveform via EDR or a sinusoidal fallback."""
        n_out = self.schema.resp_samples
        fs_out = self.schema.resp_fs

        if self.cfg.resp_method == "edr":
            edr = self._edr(ecg)
            if edr is not None:
                # Resample EDR (sampled at ECG fs) down to RESP fs.
                edr_low = self._lowpass(edr, self.schema.ecg_fs, 1.0)
                resp = self._resample(edr_low, self.schema.ecg_fs, fs_out, n_out)
                noise = self.rng.normal(
                    0.0, self.cfg.resp_noise_std, size=n_out
                )
                resp = resp + noise.astype(np.float32)
                amp = np.ptp(resp)
                if amp > 0:
                    resp = resp / amp
                return resp.astype(np.float32), METHOD_DERIVED_FROM_ECG, "EDR (R-peak amplitude modulation)"

        # Sinusoidal fallback
        t = np.arange(n_out) / fs_out
        f = self.cfg.resp_default_rate_brpm / 60.0
        phase = float(self.rng.uniform(0, 2 * np.pi))
        resp = np.cos(2 * np.pi * f * t + phase)
        noise = self.rng.normal(0.0, self.cfg.resp_noise_std, size=n_out)
        resp = resp + noise
        return resp.astype(np.float32), METHOD_RULE_BASED, f"sinusoidal {self.cfg.resp_default_rate_brpm} brpm"

    def _edr(self, ecg: np.ndarray) -> Optional[np.ndarray]:
        """Estimate ECG-derived respiration from R-peak amplitudes.

        Returns ``None`` if too few R-peaks are detected to interpolate.
        """
        peaks = detect_r_peaks(ecg, self.schema.ecg_fs)
        if peaks.size < 4:
            return None
        amps = ecg[peaks]
        # Linear interpolate amplitude trace onto the full ECG-fs grid.
        t_grid = np.arange(ecg.size, dtype=np.float64)
        return np.interp(t_grid, peaks.astype(np.float64), amps).astype(np.float64)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
        nyq = 0.5 * fs
        cutoff = min(cutoff, 0.45 * nyq)
        sos = sp_signal.butter(4, cutoff / nyq, btype="low", output="sos")
        return sp_signal.sosfiltfilt(sos, x)

    @staticmethod
    def _resample(x: np.ndarray, fs_in: float, fs_out: float, n_out: int) -> np.ndarray:
        # Lazy import to avoid a circular dependency.
        from .resampler import Resampler
        return Resampler.resample_to_length(x, fs_in, fs_out, n_out)
