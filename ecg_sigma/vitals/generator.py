"""Vitals generator.

Produces a dict ``{name: Vital}`` matching the strict on-disk schema:

  * Each ``Vital`` carries a *current* numeric value, its units, an epoch
    timestamp, and a JSON-able ``extras`` dict.

  * ``extras`` for the seven standard vitals contains
    ``{"upper_threshold", "lower_threshold", "alarm_enabled", "history"}``.

  * ``XL_Posture.extras`` instead contains
    ``{"step_count", "time_since_posture_change", "history"}``.

  * ``history`` is an ascending list of ``{"value", "timestamp"}`` entries
    covering up to ``max_vital_history`` samples. Per-vital sampling
    intervals follow the schema's history-interval table.

The generator is fully deterministic given a seeded RNG, so identical
``(seed, patient_id, event)`` inputs always yield identical extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..conditions import (
    AFIB,
    BRADYCARDIA,
    MI,
    NORMAL_SINUS,
    OTHER,
    PAUSE,
    TACHYCARDIA,
    VFIB,
    VTACH,
)


# --------------------------------------------------------------------------- #
# Per-vital profile table.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VitalProfile:
    """Schema-derived profile for one vital.

    ``value_range`` is the documented "typical range" from the schema
    table. The generator clips history samples to this range expanded by
    50 % on each side -- enough headroom for jitter at the trend
    extremes while still rejecting catastrophic outliers. The validator
    uses the same expansion so the two cannot drift.
    """

    units: str
    is_int: bool                            # True for HR and XL_Posture
    history_interval_s: Tuple[float, float] # uniform draw within this band
    default_thresholds: Optional[Tuple[float, float]]   # (low, high) or None
    value_range: Tuple[float, float]        # typical range, per the spec


VITAL_PROFILES: Dict[str, VitalProfile] = {
    "HR":         VitalProfile("bpm",          True,  (60.0, 300.0),  (50.0, 110.0), (40.0, 180.0)),
    "Pulse":      VitalProfile("bpm",          False, (60.0, 300.0),  (50.0, 110.0), (40.0, 180.0)),
    "SpO2":       VitalProfile("%",            False, (30.0, 180.0),  (90.0, 100.0), (88.0, 100.0)),
    "Systolic":   VitalProfile("mmHg",         False, (120.0, 1800.0),(90.0, 160.0), (100.0, 180.0)),
    "Diastolic":  VitalProfile("mmHg",         False, (120.0, 1800.0),(50.0, 100.0), (60.0, 110.0)),
    "RespRate":   VitalProfile("breaths/min",  False, (60.0, 600.0),  (8.0, 30.0),   (12.0, 30.0)),
    "Temp":       VitalProfile("F",            False, (300.0, 3600.0),(96.0, 101.0), (96.0, 101.0)),
    "XL_Posture": VitalProfile("degrees",      True,  (10.0, 60.0),    None,         (-10.0, 45.0)),
}

REQUIRED_VITAL_NAMES: Tuple[str, ...] = tuple(VITAL_PROFILES.keys())


def soft_value_range(profile: VitalProfile) -> Tuple[float, float]:
    """Range expanded by 50 % on each side. Single source of truth used
    by both the history generator (clip target) and the post-write
    validator (acceptance bound)."""
    lo, hi = profile.value_range
    span = hi - lo
    return (lo - 0.5 * span, hi + 0.5 * span)


# Tilt-angle ranges per posture label, in degrees (accelerometer-style).
_POSTURE_DEGREES: Dict[str, Tuple[int, int]] = {
    "Standing": (-5, 10),
    "Sitting":  (10, 25),
    "Supine":   (25, 45),
    "Prone":    (25, 45),
    "LeftLat":  (20, 40),
    "RightLat": (20, 40),
}


# --------------------------------------------------------------------------- #
# Public dataclasses.
# --------------------------------------------------------------------------- #
@dataclass
class Vital:
    """One vital-sign reading.

    ``extras`` is a JSON-able dict whose schema depends on the vital --
    see module docstring.
    """

    value: float                          # int values come through as float and
    units: str                            # the writer coerces back to int per the profile
    timestamp: float                      # epoch seconds
    extras: Dict[str, Any]


@dataclass
class VitalsConfig:
    """Per-vital configuration.

    Fields without explicit defaults fall back to :data:`VITAL_PROFILES`
    and the built-in condition baselines.
    """

    # Condition-keyed range tables for the *current* (event-time) value.
    spo2: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    bp_systolic: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    bp_diastolic_offset: Tuple[float, float] = (55.0, 85.0)
    temp_f: Tuple[float, float] = (97.5, 99.5)

    # Posture labels are deterministic-categorical; the writer rounds to int degrees.
    postures: Tuple[str, ...] = ("Supine", "LeftLat", "RightLat", "Prone", "Sitting", "Standing")

    # Optional per-vital threshold override. If a vital is absent here we
    # use ``VitalProfile.default_thresholds``.
    thresholds: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    # Posture-only extras.
    step_count_range: Tuple[int, int] = (0, 3000)
    time_since_posture_change_range: Tuple[int, int] = (0, 3600)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class VitalsGenerator:
    """Stateless. ``rng`` is provided per event for determinism."""

    def __init__(self, cfg: VitalsConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Top-level
    # ------------------------------------------------------------------ #
    def generate(
        self,
        condition: str,
        hr_bpm: float,
        resp_rate_brpm: float,
        event_timestamp_epoch: float,
        rng: np.random.Generator,
        max_vital_history: int = 30,
    ) -> Dict[str, Vital]:
        ts = float(event_timestamp_epoch)

        # ---- current values --------------------------------------------------
        hr = float(np.clip(hr_bpm, 25.0, 250.0))
        pulse = float(np.clip(hr + float(rng.normal(0.0, 1.0)), 25.0, 250.0))

        spo2_lo, spo2_hi = self._range(self.cfg.spo2, condition, default=(94.0, 99.0))
        spo2 = float(rng.uniform(spo2_lo, spo2_hi))

        sys_lo, sys_hi = self._range(self.cfg.bp_systolic, condition, default=(105.0, 135.0))
        bp_sys = float(rng.uniform(sys_lo, sys_hi))
        bp_dia = float(np.clip(
            bp_sys - rng.uniform(*self.cfg.bp_diastolic_offset),
            30.0, bp_sys - 10.0,
        ))

        rr = float(np.clip(resp_rate_brpm, 4.0, 45.0))
        temp = float(rng.uniform(*self.cfg.temp_f))

        posture_label = str(rng.choice(self.cfg.postures))
        posture_lo, posture_hi = _POSTURE_DEGREES.get(posture_label, (-10, 45))
        posture_deg = int(rng.integers(posture_lo, posture_hi + 1))

        # ---- assemble output -------------------------------------------------
        vitals: Dict[str, Vital] = {}
        current = {
            "HR":         hr,
            "Pulse":      pulse,
            "SpO2":       spo2,
            "Systolic":   bp_sys,
            "Diastolic":  bp_dia,
            "RespRate":   rr,
            "Temp":       temp,
            "XL_Posture": float(posture_deg),
        }

        for name in REQUIRED_VITAL_NAMES:
            profile = VITAL_PROFILES[name]
            val = current[name]
            baseline = self._baseline_for(name, condition, val, rng)
            history = self._make_history(
                profile, val, baseline, ts, max_vital_history, rng,
            )
            if name == "XL_Posture":
                extras = {
                    "step_count": int(rng.integers(*self.cfg.step_count_range)),
                    "time_since_posture_change": int(rng.integers(
                        *self.cfg.time_since_posture_change_range
                    )),
                    "history": history,
                }
            else:
                lo, hi = self.cfg.thresholds.get(name, profile.default_thresholds)
                extras = {
                    "upper_threshold": float(hi),
                    "lower_threshold": float(lo),
                    "alarm_enabled": True,
                    "history": history,
                }
            vitals[name] = Vital(
                value=val,
                units=profile.units,
                timestamp=ts,
                extras=extras,
            )
        return vitals

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #
    def _make_history(
        self,
        profile: VitalProfile,
        current_value: float,
        baseline: float,
        current_ts: float,
        max_points: int,
        rng: np.random.Generator,
    ) -> list:
        """Generate up to ``max_points`` ascending {"value","timestamp"} entries.

        The first sample sits at ``baseline``; subsequent samples interpolate
        linearly toward ``current_value``. Each sample is jittered to break
        the perfectly-straight trend that would otherwise look synthetic.
        Output is sorted ascending by timestamp; the most recent sample is
        one ``interval`` *before* ``current_ts`` -- the current value itself
        is the live reading and lives outside ``history``.
        """
        if max_points <= 0:
            return []
        # Pick a fixed interval for the whole history; intervals across
        # different events legitimately vary, but within an event the
        # downstream MEWS scorer expects a constant cadence.
        interval = float(rng.uniform(*profile.history_interval_s))
        # Sometimes generate slightly fewer than max for variety.
        n = int(rng.integers(max(5, max_points - 5), max_points + 1))

        # Per-sample jitter scaled by the trend amplitude so quiet vitals
        # (Temp) stay close to their baseline while noisy ones (HR) wobble.
        delta = current_value - baseline
        jitter_scale = max(abs(delta) * 0.10, 0.5)

        # Clip target. We use the validator's soft range so the generator
        # cannot produce a sample the validator would reject.
        soft_lo, soft_hi = soft_value_range(profile)

        samples = []
        for i in range(n):
            # i = 0 is OLDEST (n intervals before now); i = n-1 is NEWEST
            # (1 interval before now).
            seconds_back = (n - i) * interval
            ts = current_ts - seconds_back
            progress = (i + 1) / (n + 1)            # 0 < progress < 1
            interp = baseline + progress * delta
            jitter = float(rng.normal(0.0, jitter_scale))
            v = interp + jitter
            v = min(soft_hi, max(soft_lo, float(v)))
            if profile.is_int:
                v = int(round(v))
            else:
                v = float(round(float(v), 3))
            samples.append({"value": v, "timestamp": float(round(ts, 3))})

        # Already monotonically ascending by construction, but assert sort
        # to defend against future refactors.
        samples.sort(key=lambda s: s["timestamp"])
        return samples

    @staticmethod
    def _baseline_for(
        name: str, condition: str, current: float, rng: np.random.Generator,
    ) -> float:
        """Return a *plausible historical* baseline for the given vital.

        For alarm-causing conditions the baseline sits on the opposite side
        of normal so the trend leads "into" the alarm event. Stable
        conditions get a small Gaussian wobble around the current value.
        """
        # HR / Pulse
        if name in ("HR", "Pulse"):
            if condition == BRADYCARDIA:
                return float(current + rng.uniform(15, 35))   # was higher
            if condition in (TACHYCARDIA, VTACH, VFIB):
                return float(current - rng.uniform(15, 40))   # was lower
            if condition == AFIB:
                return float(current + rng.normal(0, 8))
            return float(current + rng.normal(0, 4))

        if name == "SpO2":
            if condition in (VTACH, VFIB, MI, PAUSE):
                return float(min(100.0, current + rng.uniform(2.0, 5.0)))
            return float(current + rng.normal(0, 0.5))

        if name == "Systolic":
            if condition in (VTACH, VFIB):
                return float(current + rng.uniform(15, 35))
            if condition in (BRADYCARDIA, MI):
                return float(current + rng.uniform(5, 15))
            return float(current + rng.normal(0, 4))

        if name == "Diastolic":
            if condition in (VTACH, VFIB):
                return float(current + rng.uniform(8, 20))
            return float(current + rng.normal(0, 3))

        if name == "RespRate":
            if condition in (VTACH, VFIB, MI):
                return float(current - rng.uniform(2, 5))
            if condition == BRADYCARDIA:
                return float(current + rng.uniform(2, 5))
            return float(current + rng.normal(0, 1))

        if name == "Temp":
            return float(current + rng.normal(0, 0.2))

        if name == "XL_Posture":
            # Posture frequently changes; baseline is just another posture.
            return float(rng.uniform(-10, 45))

        return float(current)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _range(
        table: Mapping[str, Tuple[float, float]],
        condition: str,
        default: Tuple[float, float],
    ) -> Tuple[float, float]:
        if condition in table:
            return tuple(table[condition])
        if OTHER in table:
            return tuple(table[OTHER])
        return default
