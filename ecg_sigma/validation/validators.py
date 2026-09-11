"""Pre-write payload validators and post-write file inspectors.

Two layers:

1. :func:`validate_event_payload` -- runs before we write a single event.
   Catches programmer errors (wrong shape, missing leads) early.

2. :func:`validate_pipeline_output` -- re-opens a written file and asserts
   the strict on-disk schema, including:
     * Pacer-only ``ecg/extras`` JSON.
     * Empty ``ppg/extras`` and ``resp/extras`` JSON dicts.
     * Vitals ``extras`` shape: thresholds + alarm_enabled + history for
       the seven standard vitals; step_count + time_since_posture_change
       + history for XL_Posture.
     * Optional history-integrity check (``verify_history=True``) -- sort
       order, sample count, value bounds.

   It also gates on *content*, not just structure: no flat/zero-filled
   waveform, every ``condition`` inside the unified vocabulary, every
   waveform carrying its ``source``/``method`` provenance attributes, and
   ``heart_rate`` agreeing with ``vitals/HR/value``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from ..conditions import ALL_CONDITIONS
from ..schema import (
    DEFAULT_LEADS,
    SCHEMA,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    SchemaSpec,
    required_window_lengths,
)
from ..vitals.generator import VITAL_PROFILES, soft_value_range


class ValidationError(Exception):
    """Raised when a payload or output file violates the schema."""


REQUIRED_METADATA_ATTRS = (
    "patient_id",
    "sampling_rate_ecg",
    "sampling_rate_ppg",
    "sampling_rate_resp",
    "alarm_time_epoch",
    "alarm_offset_seconds",
    "seconds_before_event",
    "seconds_after_event",
    "data_quality_score",
    "device_info",
    "max_vital_history",
)

REQUIRED_VITALS = (
    "HR", "Pulse", "SpO2", "Systolic", "Diastolic",
    "RespRate", "Temp", "XL_Posture",
)
STANDARD_VITALS = tuple(v for v in REQUIRED_VITALS if v != "XL_Posture")
INT_VITALS = ("HR", "XL_Posture")

# Provenance attributes every waveform dataset must carry so a consumer can
# tell real signal from synthesis without re-running the pipeline.
REQUIRED_PROVENANCE_ATTRS = ("source", "method")
VALID_SOURCES = (SOURCE_REAL, SOURCE_SYNTHETIC)

# A waveform whose peak-to-peak span is below this is flat: a dead channel,
# a zero-fill, or a saturated segment. Never a usable ECG/PPG/RESP trace.
FLATLINE_PTP_EPS = 1e-9

# ``heart_rate`` (group attr, float bpm) and ``vitals/HR/value`` (int bpm)
# come from one measurement; the generator clips to this band before
# rounding, so they may differ by at most one bpm of rounding.
_HR_CLIP = (25.0, 250.0)
_HR_TOLERANCE_BPM = 1.0

# Soft floor for /metadata data_quality_score. Empirically the in-band SQI
# runs ~0.38-0.49 on clean MIT-BIH records and ~0.10-0.34 on the ones
# PhysioNet flags as noisy, so this floor picks out the worst offenders
# (e.g. records 203, 207) without failing them.
LOW_QUALITY_WARN_BELOW = 0.20


# --------------------------------------------------------------------------- #
# Pre-write
# --------------------------------------------------------------------------- #
def validate_event_payload(evt, schema: SchemaSpec = SCHEMA) -> None:
    """Validate an :class:`EventPayload` before it goes to disk."""
    from ..writers.hdf5_writer import EventPayload

    if not isinstance(evt, EventPayload):
        raise ValidationError(f"expected EventPayload, got {type(evt)}")

    if not evt.condition:
        raise ValidationError("event has empty condition")
    if not math.isfinite(evt.heart_rate_bpm):
        raise ValidationError(f"heart_rate not finite: {evt.heart_rate_bpm}")
    if evt.event_timestamp_epoch < 0:
        raise ValidationError(f"event_timestamp_epoch < 0: {evt.event_timestamp_epoch}")

    needed = required_window_lengths(schema)

    missing = [l for l in DEFAULT_LEADS if l not in evt.ecg]
    if missing:
        raise ValidationError(f"ECG missing leads: {missing}")
    for lead, (sig, _tag) in evt.ecg.items():
        _check_signal(f"ecg/{lead}", sig, needed["ecg"])

    _check_signal("ppg/PPG", evt.ppg[0], needed["ppg"])
    _check_signal("resp/RESP", evt.resp[0], needed["resp"])

    missing_v = set(REQUIRED_VITALS) - set(evt.vitals.keys())
    if missing_v:
        raise ValidationError(f"vitals missing keys: {sorted(missing_v)}")

    # Pacer offset must lie inside the ECG window.
    if not (0 <= int(evt.pacer_offset) < needed["ecg"]):
        raise ValidationError(
            f"pacer_offset {evt.pacer_offset} outside [0, {needed['ecg']})"
        )


def _check_signal(name: str, sig: np.ndarray, expected_len: int) -> None:
    sig = np.asarray(sig)
    if sig.ndim != 1:
        raise ValidationError(f"{name}: expected 1-D, got shape {sig.shape}")
    if sig.size != expected_len:
        raise ValidationError(
            f"{name}: expected {expected_len} samples, got {sig.size}"
        )
    if not np.all(np.isfinite(sig)):
        raise ValidationError(f"{name}: contains non-finite values")
    ptp = float(np.ptp(sig))
    if ptp < FLATLINE_PTP_EPS:
        raise ValidationError(
            f"{name}: flat channel (peak-to-peak {ptp:g}); refusing to write "
            "a constant waveform"
        )


# --------------------------------------------------------------------------- #
# Post-write
# --------------------------------------------------------------------------- #
def validate_pipeline_output(
    path: str,
    schema: SchemaSpec = SCHEMA,
    *,
    max_events: int = 0,
    verify_history: bool = True,
) -> List[str]:
    """Re-open a written file and assert the strict schema.

    Returns soft warnings (e.g. unusually low quality score). Raises
    :class:`ValidationError` for any structural violation.

    Parameters
    ----------
    verify_history:
        When True (default), additionally checks the per-vital
        ``extras.history`` array: ascending timestamps, sample-count
        within ``max_vital_history``, and values within the documented
        clinical range.
    """
    if not os.path.exists(path):
        raise ValidationError(f"file not found: {path}")
    needed = required_window_lengths(schema)
    warnings: List[str] = []

    with h5py.File(path, "r") as f:
        if "metadata" not in f:
            raise ValidationError("/metadata group missing")
        md = f["metadata"].attrs

        for k in REQUIRED_METADATA_ATTRS:
            if k not in md:
                raise ValidationError(f"/metadata missing attr {k!r}")

        quality = float(md["data_quality_score"])
        if not (0.0 <= quality <= 1.0):
            raise ValidationError(
                f"/metadata data_quality_score {quality} outside [0, 1]"
            )
        if quality < LOW_QUALITY_WARN_BELOW:
            warnings.append(
                f"data_quality_score {quality:.3f} is below "
                f"{LOW_QUALITY_WARN_BELOW}; the source signal is likely noisy"
            )

        max_vital_history = int(md["max_vital_history"])
        if max_vital_history <= 0:
            raise ValidationError(f"/metadata max_vital_history invalid: {max_vital_history}")

        for k, expected in (
            ("sampling_rate_ecg",  schema.ecg_fs),
            ("sampling_rate_ppg",  schema.ppg_fs),
            ("sampling_rate_resp", schema.resp_fs),
        ):
            if abs(float(md[k]) - expected) > 1e-3:
                raise ValidationError(
                    f"/metadata {k}={float(md[k])} != {expected}"
                )
        if abs(float(md["alarm_offset_seconds"])
               - float(schema.seconds_before_event)) > 1e-6:
            raise ValidationError("/metadata alarm_offset_seconds mismatch")

        events = sorted(k for k in f.keys() if k.startswith("event_"))
        if max_events:
            events = events[:max_events]
        if not events:
            warnings.append("file contains no events")

        for ev in events:
            grp = f[ev]
            for attr in ("condition", "heart_rate", "event_timestamp"):
                if attr not in grp.attrs:
                    raise ValidationError(f"{ev} missing attr {attr!r}")

            condition = _attr_to_str(grp.attrs["condition"])
            if condition not in ALL_CONDITIONS:
                raise ValidationError(
                    f"{ev} condition {condition!r} is not in the unified "
                    f"vocabulary {sorted(ALL_CONDITIONS)}"
                )

            if "timestamp" not in grp:
                raise ValidationError(f"{ev}/timestamp dataset missing")
            if "uuid" not in grp:
                raise ValidationError(f"{ev}/uuid dataset missing")
            if grp["timestamp"].dtype.kind != "f":
                raise ValidationError(
                    f"{ev}/timestamp must be float, got dtype {grp['timestamp'].dtype}"
                )

            # ECG: signals + pacer extras.
            for lead in DEFAULT_LEADS:
                _check_dataset(grp, f"ecg/{lead}", needed["ecg"], ev)
            _check_pacer_extras(grp, ev, needed["ecg"])

            # PPG / RESP: signals + empty extras.
            _check_dataset(grp, "ppg/PPG", needed["ppg"], ev)
            _check_empty_extras(grp, "ppg/extras", ev)
            _check_dataset(grp, "resp/RESP", needed["resp"], ev)
            _check_empty_extras(grp, "resp/extras", ev)

            # Vitals.
            vitals = grp.get("vitals")
            if vitals is None:
                raise ValidationError(f"{ev}/vitals missing")
            for required in REQUIRED_VITALS:
                if required not in vitals:
                    raise ValidationError(f"{ev}/vitals/{required} missing")
                vg = vitals[required]
                for child in ("value", "units", "timestamp", "extras"):
                    if child not in vg:
                        raise ValidationError(
                            f"{ev}/vitals/{required}/{child} missing"
                        )
                if vg["timestamp"].dtype.kind != "f":
                    raise ValidationError(
                        f"{ev}/vitals/{required}/timestamp must be float"
                    )
                if required in INT_VITALS and vg["value"].dtype.kind not in ("i", "u"):
                    raise ValidationError(
                        f"{ev}/vitals/{required}/value must be int, "
                        f"got dtype {vg['value'].dtype}"
                    )
                extras = _read_json_dataset(vg["extras"], f"{ev}/vitals/{required}/extras")
                _check_vital_extras_shape(required, extras, ev)
                if verify_history:
                    _check_history(
                        required, extras["history"], max_vital_history, ev,
                    )

            _check_heart_rate_agreement(grp, ev)

    return warnings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_dataset(grp, path: str, expected_len: int, ev_name: str) -> None:
    """Shape/dtype contract plus content and provenance gates.

    Structure alone cannot tell a real waveform from a zero-fill, so this
    also rejects flat channels and requires the ``source``/``method``
    provenance attributes that carry the real-vs-synthetic answer.
    """
    ds = grp.get(path)
    if ds is None:
        raise ValidationError(f"{ev_name}/{path} missing")
    if ds.shape != (expected_len,):
        raise ValidationError(
            f"{ev_name}/{path} shape {ds.shape} != ({expected_len},)"
        )
    if ds.dtype != np.float32:
        raise ValidationError(f"{ev_name}/{path} dtype {ds.dtype} != float32")

    data = ds[...]
    if not np.all(np.isfinite(data)):
        raise ValidationError(f"{ev_name}/{path} contains non-finite values")
    ptp = float(np.ptp(data))
    if ptp < FLATLINE_PTP_EPS:
        raise ValidationError(
            f"{ev_name}/{path} is flat (peak-to-peak {ptp:g}); a constant "
            "channel is a zero-fill or dead lead, not a signal"
        )

    _check_provenance(ds, f"{ev_name}/{path}")


def _check_provenance(ds, label: str) -> None:
    """Every waveform must declare where it came from."""
    for attr in REQUIRED_PROVENANCE_ATTRS:
        if attr not in ds.attrs:
            raise ValidationError(f"{label} missing provenance attr {attr!r}")
    source = _attr_to_str(ds.attrs["source"])
    if source not in VALID_SOURCES:
        raise ValidationError(
            f"{label} source {source!r} not one of {list(VALID_SOURCES)}"
        )
    if not _attr_to_str(ds.attrs["method"]).strip():
        raise ValidationError(f"{label} has an empty 'method' provenance attr")


def _check_heart_rate_agreement(grp, ev_name: str) -> None:
    """``heart_rate`` group attr must match ``vitals/HR/value``.

    They are two renderings of one measurement; a mismatch means the
    vitals block drifted away from the ECG it is supposed to describe.
    """
    attr_hr = float(grp.attrs["heart_rate"])
    if not math.isfinite(attr_hr):
        raise ValidationError(f"{ev_name} heart_rate attr not finite: {attr_hr}")
    vital_hr = float(np.array(grp["vitals/HR/value"]))
    expected = round(min(max(attr_hr, _HR_CLIP[0]), _HR_CLIP[1]))
    if abs(expected - vital_hr) > _HR_TOLERANCE_BPM:
        raise ValidationError(
            f"{ev_name} heart_rate attr {attr_hr:.2f} bpm disagrees with "
            f"vitals/HR/value {vital_hr:.0f} bpm (expected ~{expected})"
        )


def _check_pacer_extras(grp, ev_name: str, n_ecg_samples: int) -> None:
    ds = grp.get("ecg/extras")
    if ds is None:
        raise ValidationError(f"{ev_name}/ecg/extras missing")
    payload = _read_json_dataset(ds, f"{ev_name}/ecg/extras")
    if not isinstance(payload, dict):
        raise ValidationError(
            f"{ev_name}/ecg/extras is not a JSON object"
        )
    for required in ("pacer_info", "pacer_offset"):
        if required not in payload:
            raise ValidationError(
                f"{ev_name}/ecg/extras missing field {required!r}"
            )
    pacer_info = int(payload["pacer_info"])
    pacer_offset = int(payload["pacer_offset"])
    if pacer_info < 0 or pacer_info > 0xFFFFFFFF:
        raise ValidationError(
            f"{ev_name}/ecg/extras pacer_info out of 32-bit range: {pacer_info}"
        )
    if not (0 <= pacer_offset < n_ecg_samples):
        raise ValidationError(
            f"{ev_name}/ecg/extras pacer_offset {pacer_offset} "
            f"outside [0, {n_ecg_samples})"
        )


def _check_empty_extras(grp, path: str, ev_name: str) -> None:
    ds = grp.get(path)
    if ds is None:
        raise ValidationError(f"{ev_name}/{path} missing")
    payload = _read_json_dataset(ds, f"{ev_name}/{path}")
    if payload != {}:
        raise ValidationError(
            f"{ev_name}/{path} expected empty JSON object, got {payload!r}"
        )


def _check_vital_extras_shape(name: str, extras: Any, ev_name: str) -> None:
    if not isinstance(extras, dict):
        raise ValidationError(
            f"{ev_name}/vitals/{name}/extras is not a JSON object"
        )
    if "history" not in extras:
        raise ValidationError(
            f"{ev_name}/vitals/{name}/extras missing 'history' key"
        )

    if name == "XL_Posture":
        for required in ("step_count", "time_since_posture_change"):
            if required not in extras:
                raise ValidationError(
                    f"{ev_name}/vitals/{name}/extras missing {required!r}"
                )
        if not isinstance(extras["step_count"], int):
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras step_count must be int"
            )
    else:
        for required in ("upper_threshold", "lower_threshold", "alarm_enabled"):
            if required not in extras:
                raise ValidationError(
                    f"{ev_name}/vitals/{name}/extras missing {required!r}"
                )
        if extras["alarm_enabled"] is not True:
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras alarm_enabled must be True for "
                "standard vitals"
            )
        lo = float(extras["lower_threshold"])
        hi = float(extras["upper_threshold"])
        if not (lo < hi):
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras invalid thresholds "
                f"(lower={lo}, upper={hi})"
            )


def _check_history(
    name: str, history: Any, max_points: int, ev_name: str,
) -> None:
    if not isinstance(history, list):
        raise ValidationError(
            f"{ev_name}/vitals/{name}/extras/history must be a list"
        )
    if len(history) > max_points:
        raise ValidationError(
            f"{ev_name}/vitals/{name}/extras/history has "
            f"{len(history)} samples > max {max_points}"
        )
    last_ts = -float("inf")
    profile = VITAL_PROFILES.get(name)
    if profile is None:
        val_lo_soft, val_hi_soft = -1e18, 1e18
    else:
        val_lo_soft, val_hi_soft = soft_value_range(profile)
    for i, sample in enumerate(history):
        if not isinstance(sample, dict):
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras/history[{i}] not a dict"
            )
        if "value" not in sample or "timestamp" not in sample:
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras/history[{i}] missing fields"
            )
        ts = float(sample["timestamp"])
        if ts < last_ts:
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras/history not sorted ascending"
                f" at index {i}"
            )
        last_ts = ts
        v = float(sample["value"])
        if not (val_lo_soft <= v <= val_hi_soft):
            raise ValidationError(
                f"{ev_name}/vitals/{name}/extras/history[{i}].value={v} "
                f"outside soft range [{val_lo_soft}, {val_hi_soft}]"
            )


# --------------------------------------------------------------------------- #
# Low-level
# --------------------------------------------------------------------------- #
def _read_json_dataset(ds, label: str) -> Any:
    try:
        return json.loads(_attr_to_str(np.array(ds)))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{label} not valid JSON: {exc}") from exc


def _attr_to_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "O"):
            return value.tobytes().decode("utf-8")
        return str(value)
    return str(value)
