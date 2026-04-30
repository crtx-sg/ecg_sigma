"""Pacer metadata generation for the ECG ``extras`` block.

The schema mandates ``ecg/extras`` to carry exactly two fields::

    {
        "pacer_info":   <int>,        # bit-packed (type|rate<<8|amp<<16|flags<<24)
        "pacer_offset": <int>          # 0 .. n_samples-1 within the 12 s ECG window
    }

Pacer presence and offset placement are condition-driven:

  * VTACH / VFIB        -- ~40 % chance of pacer; bimodal early/late offset.
  * BRADYCARDIA         -- ~80 % chance of pacer; bimodal early/late offset.
  * Everything else     -- ~5 % chance of pacer; uniform 20-80 % offset.

When no pacer is generated, both fields are written as 0, matching the
"no pacer" decoding rule (``pacer_type = pacer_info & 0xFF == 0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import numpy as np

from ..conditions import BRADYCARDIA, VFIB, VTACH


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PacerConfig:
    """Tunables for pacer synthesis. All defaults match the schema spec."""

    # Probability of pacer-on, by condition. Anything missing falls back to
    # ``default_probability``.
    probabilities: Mapping[str, float] = field(default_factory=lambda: {
        VTACH:       0.40,
        VFIB:        0.40,
        BRADYCARDIA: 0.80,
    })
    default_probability: float = 0.05

    # Pacer parameter ranges (inclusive bounds for integer draw).
    type_min: int = 1          # 1=Single, 2=Dual, 3=Biventricular
    type_max: int = 3
    rate_min_bpm: int = 60
    rate_max_bpm: int = 100
    amplitude_min: int = 1
    amplitude_max: int = 10

    # Offset windows expressed as fractions of the ECG window.
    bimodal_conditions: Tuple[str, ...] = (VTACH, VFIB, BRADYCARDIA)
    bimodal_early_frac: Tuple[float, float] = (0.10, 0.25)
    bimodal_late_frac:  Tuple[float, float] = (0.75, 0.90)
    uniform_frac:       Tuple[float, float] = (0.20, 0.80)


# --------------------------------------------------------------------------- #
# Encoders / decoders -- public so tests and downstream tools can re-use them.
# --------------------------------------------------------------------------- #
def pack_pacer_info(pacer_type: int, rate_bpm: int, amplitude: int,
                    flags: int = 0) -> int:
    """Pack the four byte-fields into a single 32-bit integer."""
    for name, val, lo, hi in (
        ("pacer_type", pacer_type, 0, 0xFF),
        ("rate_bpm",   rate_bpm,   0, 0xFF),
        ("amplitude",  amplitude,  0, 0xFF),
        ("flags",      flags,      0, 0xFF),
    ):
        if not (lo <= int(val) <= hi):
            raise ValueError(f"{name} {val} out of byte range [{lo},{hi}]")
    return (
        int(pacer_type) & 0xFF
        | (int(rate_bpm) & 0xFF) << 8
        | (int(amplitude) & 0xFF) << 16
        | (int(flags) & 0xFF) << 24
    )


def unpack_pacer_info(pacer_info: int) -> dict:
    """Inverse of :func:`pack_pacer_info`. Useful for tests/inspect tools."""
    pi = int(pacer_info)
    return {
        "pacer_type": pi & 0xFF,
        "rate_bpm":   (pi >> 8) & 0xFF,
        "amplitude":  (pi >> 16) & 0xFF,
        "flags":      (pi >> 24) & 0xFF,
    }


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class PacerGenerator:
    """Stateless generator. Pass an RNG per-event for determinism."""

    def __init__(self, cfg: PacerConfig = PacerConfig()):
        self.cfg = cfg

    def generate(
        self,
        condition: str,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[int, int]:
        """Return ``(pacer_info, pacer_offset)`` for one event.

        Parameters
        ----------
        condition:
            Unified condition label (e.g. ``"VTACH"``).
        n_samples:
            ECG window length in samples (typically 2400 for 12 s @ 200 Hz).
        rng:
            NumPy generator. The same ``(seed, patient_id, event_idx)`` should
            yield the same pacer every run.
        """
        if n_samples <= 0:
            return 0, 0

        prob = self.cfg.probabilities.get(condition, self.cfg.default_probability)
        if float(rng.random()) >= prob:
            return 0, 0

        pacer_type = int(rng.integers(self.cfg.type_min, self.cfg.type_max + 1))
        rate = int(rng.integers(self.cfg.rate_min_bpm, self.cfg.rate_max_bpm + 1))
        amp = int(rng.integers(self.cfg.amplitude_min, self.cfg.amplitude_max + 1))
        flags = 0
        info = pack_pacer_info(pacer_type, rate, amp, flags)
        offset = self._pick_offset(condition, n_samples, rng)
        return info, offset

    def _pick_offset(
        self, condition: str, n_samples: int, rng: np.random.Generator,
    ) -> int:
        if condition in self.cfg.bimodal_conditions:
            if float(rng.random()) < 0.5:
                lo, hi = self.cfg.bimodal_early_frac
            else:
                lo, hi = self.cfg.bimodal_late_frac
        else:
            lo, hi = self.cfg.uniform_frac
        return int(rng.uniform(lo * n_samples, hi * n_samples))
