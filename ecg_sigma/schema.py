"""Constants and dataclasses describing the canonical output schema.

The output is a per-patient, per-month HDF5 file:

    {patient_id}_{YYYY-MM}.h5
    ├── /metadata                         (group, attributes only)
    └── /event_XXXX                       (group, one per event)
        ├── @condition, @heart_rate, @event_timestamp, @alarm_time_epoch
        ├── /timestamp                    (scalar int64, epoch seconds)
        ├── /uuid                         (scalar str)
        ├── /ecg                          (group)
        │     ├── ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX  (1-D float32)
        ├── /ppg                          (group)
        │     └── pleth                   (1-D float32)
        ├── /resp                         (group)
        │     └── waveform                (1-D float32)
        └── /vitals                       (group)
              └── {NAME}                  (group, attrs: value/units/timestamp/extras)

All sampling rates and window lengths are fixed by :class:`SchemaSpec`.
The single source of truth for these constants lives here; downstream
modules import them rather than hard-coding numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# Canonical lead order written to /event_*/ecg/.
DEFAULT_LEADS: Tuple[str, ...] = (
    "ECG1",  # Lead I
    "ECG2",  # Lead II
    "ECG3",  # Lead III
    "aVR",
    "aVL",
    "aVF",
    "vVX",   # precordial / unipolar V-equivalent
)

# Mapping ECG{1,2,3} -> standard limb-lead label, used by the lead mapper.
ECG_TO_LIMB = {"ECG1": "I", "ECG2": "II", "ECG3": "III"}

# Tag schema for `extras` JSON.
SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"
METHOD_DERIVED_FROM_ECG = "derived_from_ecg"
METHOD_RULE_BASED = "rule_based"
METHOD_DIRECT = "direct"
METHOD_EINTHOVEN = "einthoven"


@dataclass(frozen=True)
class SchemaSpec:
    """Immutable description of the target signal grid.

    Attributes
    ----------
    ecg_fs, ppg_fs, resp_fs:
        Sampling rates in Hz.
    seconds_before_event, seconds_after_event:
        Window placement around the event onset.
    """

    ecg_fs: float = 200.0
    ppg_fs: float = 75.0
    resp_fs: float = 100.0 / 3.0
    seconds_before_event: int = 6
    seconds_after_event: int = 6
    device_info: str = "RMSAI-SimDevice-v2.0"
    leads: Tuple[str, ...] = DEFAULT_LEADS

    @property
    def window_seconds(self) -> int:
        return int(self.seconds_before_event + self.seconds_after_event)

    @property
    def ecg_samples(self) -> int:
        return int(round(self.ecg_fs * self.window_seconds))

    @property
    def ppg_samples(self) -> int:
        return int(round(self.ppg_fs * self.window_seconds))

    @property
    def resp_samples(self) -> int:
        return int(round(self.resp_fs * self.window_seconds))


# Canonical instance used as a fall-back if no override is provided.
SCHEMA = SchemaSpec()


@dataclass
class ExtrasTag:
    """Helper for the `extras` JSON payload attached to every signal/vital."""

    source: str            # "real" | "synthetic"
    method: str            # "direct" | "derived_from_ecg" | "einthoven" | "rule_based"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"source": self.source, "method": self.method}
        if self.notes:
            out["notes"] = self.notes
        if self.extra:
            out.update(self.extra)
        return out


def required_window_lengths(spec: SchemaSpec = SCHEMA) -> dict:
    """Convenience: numeric expectations used by validators."""
    return {
        "ecg": spec.ecg_samples,
        "ppg": spec.ppg_samples,
        "resp": spec.resp_samples,
    }
