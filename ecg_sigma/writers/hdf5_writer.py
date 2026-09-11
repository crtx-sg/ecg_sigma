"""HDF5 writer matching the canonical ICU schema.

File layout (strict; see README and ASSUMPTIONS for details):

    <patient_id>_<YYYY-MM>.h5
    ├── /metadata                                 (group, attributes only)
    │     attrs: patient_id,
    │            sampling_rate_ecg, sampling_rate_ppg, sampling_rate_resp,
    │            alarm_time_epoch, alarm_offset_seconds,
    │            seconds_before_event, seconds_after_event,
    │            data_quality_score, device_info, max_vital_history
    └── /event_1001 ...                            (group; one per event)
          attrs: condition, heart_rate, event_timestamp        (spec'd)
                 source_label, source_sample,                  (traceability)
                 source_beat_condition, source_rhythm_condition
          /timestamp                                            (scalar float64)
          /uuid                                                 (scalar utf-8 string)
          /ecg/{ECG1,ECG2,ECG3,aVR,aVL,aVF,vVX}                 (1-D float32, gzip)
          /ecg/extras                                           (utf-8 JSON: {pacer_info, pacer_offset})
          /ppg/PPG                                              (1-D float32, gzip)
          /ppg/extras                                           (utf-8 JSON: {})
          /resp/RESP                                            (1-D float32, gzip)
          /resp/extras                                          (utf-8 JSON: {})
          /vitals/{HR,Pulse,SpO2,Systolic,Diastolic,
                   RespRate,Temp,XL_Posture}                    (group)
                value      scalar (int for HR / XL_Posture, else float64)
                units      scalar utf-8 string
                timestamp  scalar float64, epoch seconds
                extras     scalar utf-8 JSON  -- thresholds + alarm_enabled + history
                                              -- (XL_Posture: step_count + time_since_posture_change + history)

Per-lead provenance is preserved as HDF5 attributes on each lead dataset
(``source``, ``method``, ``notes``) so provenance is not lost when the
``ecg/extras`` JSON is restricted to pacer metadata.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from ..schema import (
    DEFAULT_LEADS,
    ExtrasTag,
    SchemaSpec,
)
from ..utils.logging import get_logger
from ..vitals import Vital

_log = get_logger(__name__)
_EVENT_INDEX_BASE = 1001       # event_1001, event_1002, ...

# Vitals whose ``value`` dataset must be stored as int per the schema.
_INT_VALUE_VITALS = frozenset({"HR", "XL_Posture"})


# --------------------------------------------------------------------------- #
# Public dataclasses describing the writer's payload.
# --------------------------------------------------------------------------- #
@dataclass
class EventPayload:
    """Everything needed to write a single ``/event_NNNN`` group."""

    condition: str
    source_label: str                                    # raw dataset label
    event_timestamp_epoch: float
    heart_rate_bpm: float
    data_quality_score: float                            # aggregated into /metadata
    ecg: Dict[str, Tuple[np.ndarray, ExtrasTag]]         # lead -> (signal, provenance)
    ppg: Tuple[np.ndarray, ExtrasTag]                    # provenance kept as ds-attrs
    resp: Tuple[np.ndarray, ExtrasTag]
    vitals: Dict[str, Vital]
    pacer_info: int = 0                                  # bit-packed, 0 = no pacer
    pacer_offset: int = 0                                # ECG sample index
    extras: Dict[str, Any] = field(default_factory=dict)

    # Traceability back to the source annotation. ``source_sample`` is the
    # onset index in *source-fs* coordinates, so an event can be replayed
    # against the original record.
    source_sample: int = -1
    source_beat_condition: str = ""                      # label from beat morphology
    source_rhythm_condition: str = ""                    # label from rhythm context


@dataclass
class FilePayload:
    """One HDF5 file -- one (patient, year, month) bucket of events."""

    patient_id: str
    dataset: str
    year: int
    month: int
    events: List[EventPayload]
    record_metadata: Dict[str, Any]
    max_vital_history: int = 30
    source_fs: float = 0.0                               # raw sampling rate, Hz
    source_channels: Tuple[str, ...] = ()                # raw channel names as read


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class HDF5Writer:
    """Idempotent writer.

    Calling :meth:`write` overwrites the destination file (the writer never
    silently appends). Files are placed under ``output_dir/<dataset>/`` to
    isolate runs.
    """

    def __init__(
        self,
        output_dir: str,
        schema: SchemaSpec,
        compression: str = "gzip",
        compression_opts: int = 4,
        file_pattern: str = "{patient_id}_{year}-{month:02d}.h5",
    ):
        self.output_dir = output_dir
        self.schema = schema
        self.compression = compression
        self.compression_opts = compression_opts
        self.file_pattern = file_pattern

    # ------------------------------------------------------------------ #
    # Top level
    # ------------------------------------------------------------------ #
    def write(self, payload: FilePayload) -> str:
        target_dir = os.path.join(self.output_dir, payload.dataset)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(
            target_dir,
            self.file_pattern.format(
                patient_id=payload.patient_id,
                year=payload.year,
                month=payload.month,
            ),
        )
        # Atomic write: stage to temp file, then rename.
        tmp = out_path + ".tmp"
        try:
            with h5py.File(tmp, "w") as f:
                self._write_metadata(f, payload)
                for offset, evt in enumerate(payload.events):
                    self._write_event(f, _EVENT_INDEX_BASE + offset, evt)
            os.replace(tmp, out_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        _log.info(
            "wrote %s (%d events)", os.path.basename(out_path), len(payload.events)
        )
        return os.path.abspath(out_path)

    # ------------------------------------------------------------------ #
    # /metadata
    # ------------------------------------------------------------------ #
    def _write_metadata(self, f: h5py.File, payload: FilePayload) -> None:
        meta = f.create_group("metadata")
        n = len(payload.events)
        agg_quality = (
            float(np.mean([e.data_quality_score for e in payload.events]))
            if n else 0.0
        )
        first_ts = float(payload.events[0].event_timestamp_epoch) if n else 0.0
        offset = float(self.schema.seconds_before_event)

        attrs = {
            "patient_id":           payload.patient_id,
            "sampling_rate_ecg":    float(self.schema.ecg_fs),
            "sampling_rate_ppg":    float(self.schema.ppg_fs),
            "sampling_rate_resp":   float(self.schema.resp_fs),
            "alarm_time_epoch":     first_ts,
            "alarm_offset_seconds": offset,
            "seconds_before_event": float(self.schema.seconds_before_event),
            "seconds_after_event":  float(self.schema.seconds_after_event),
            "data_quality_score":   agg_quality,
            "device_info":          self.schema.device_info,
            "max_vital_history":    int(payload.max_vital_history),
        }
        for k, v in attrs.items():
            self._set_attr(meta, k, v)

        # Source provenance. Out-of-band relative to the strict schema, but
        # without it an event cannot be traced back to the record it came
        # from -- which makes the file unauditable.
        rec_md = payload.record_metadata or {}
        # Prefer the channels actually read over whatever the loader chose
        # to record, so traceability does not depend on per-loader metadata.
        channels = payload.source_channels or rec_md.get("channels_raw") or []
        for k, v in (
            ("source_dataset",        payload.dataset),
            ("source_record_path",    str(rec_md.get("source_record_path", ""))),
            ("source_channels",       ",".join(str(c) for c in channels)),
            ("source_sampling_rate",  float(payload.source_fs)),
            ("source_n_samples",      int(rec_md.get("n_samples_raw", 0))),
        ):
            self._set_attr(meta, k, v)

    # ------------------------------------------------------------------ #
    # /event_NNNN
    # ------------------------------------------------------------------ #
    def _write_event(self, f: h5py.File, index: int, evt: EventPayload) -> None:
        grp_name = f"event_{index:04d}"
        grp = f.create_group(grp_name)

        # Group-level attributes -- the three spec'd fields ...
        self._set_attr(grp, "condition", evt.condition)
        self._set_attr(grp, "heart_rate", float(evt.heart_rate_bpm))
        self._set_attr(grp, "event_timestamp", float(evt.event_timestamp_epoch))

        # ... plus out-of-band traceability. ``condition`` is the alarm-priority
        # winner between the beat morphology and the background rhythm; both
        # inputs are kept here so a consumer can re-derive a morphology-first
        # label without re-running the pipeline.
        self._set_attr(grp, "source_label", evt.source_label)
        self._set_attr(grp, "source_sample", int(evt.source_sample))
        self._set_attr(grp, "source_beat_condition", evt.source_beat_condition)
        self._set_attr(grp, "source_rhythm_condition", evt.source_rhythm_condition)

        # Scalar datasets at the event root.
        grp.create_dataset(
            "timestamp", data=np.float64(evt.event_timestamp_epoch),
        )
        # The pipeline always supplies a deterministic uuid5; this fallback
        # keeps direct writer users reproducible too.
        evt_uuid = (evt.extras or {}).get("uuid") or str(uuid.uuid5(
            uuid.NAMESPACE_OID,
            f"{evt.condition}/{evt.event_timestamp_epoch!r}/{index}",
        ))
        grp.create_dataset("uuid", data=np.bytes_(evt_uuid.encode("utf-8")))

        # ECG: 7 leads + spec'd pacer extras. Per-lead provenance becomes
        # dataset attributes (``source``/``method``/``notes``) so callers
        # can still audit synthetic leads after the strict-extras change.
        ecg_grp = grp.create_group("ecg")
        for lead in DEFAULT_LEADS:
            sig, tag = evt.ecg[lead]
            ds = self._write_signal_dataset(ecg_grp, lead, sig)
            self._tag_provenance(ds, tag, units="mV")
        self._write_extras(ecg_grp, {
            "pacer_info":   int(evt.pacer_info),
            "pacer_offset": int(evt.pacer_offset),
        })

        # PPG: empty extras dict, provenance on the dataset.
        ppg_grp = grp.create_group("ppg")
        ppg_sig, ppg_tag = evt.ppg
        ppg_ds = self._write_signal_dataset(ppg_grp, "PPG", ppg_sig)
        self._tag_provenance(ppg_ds, ppg_tag, units="a.u.")
        self._write_extras(ppg_grp, {})

        # RESP: empty extras dict, provenance on the dataset.
        resp_grp = grp.create_group("resp")
        resp_sig, resp_tag = evt.resp
        resp_ds = self._write_signal_dataset(resp_grp, "RESP", resp_sig)
        self._tag_provenance(resp_ds, resp_tag, units="a.u.")
        self._write_extras(resp_grp, {})

        # Vitals -- each as a sub-group with value/units/timestamp/extras
        # written as *datasets*. ``extras`` here comes from the generator
        # already in dict form; the writer just JSON-serialises it.
        vitals_grp = grp.create_group("vitals")
        for name, vit in evt.vitals.items():
            self._write_vital(vitals_grp, name, vit)

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _write_signal_dataset(
        self, parent: h5py.Group, name: str, signal: np.ndarray,
    ) -> h5py.Dataset:
        """Write a 1-D float32 signal with gzip compression, return the dataset."""
        signal = np.asarray(signal, dtype=np.float32)
        return parent.create_dataset(
            name,
            data=signal,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=True,
            chunks=True,
        )

    def _tag_provenance(
        self, ds: h5py.Dataset, tag: ExtrasTag, units: str,
    ) -> None:
        """Attach optional audit attributes to a signal dataset.

        These are *not* in the strict schema but live as attributes (not
        in any spec'd ``extras``) so they cannot conflict with the
        documented JSON contract.
        """
        self._set_attr(ds, "units", units)
        self._set_attr(ds, "source", tag.source)
        self._set_attr(ds, "method", tag.method)
        if tag.notes:
            self._set_attr(ds, "notes", tag.notes)

    @staticmethod
    def _write_extras(parent: h5py.Group, payload: Dict[str, Any]) -> None:
        """Write a JSON-encoded ``extras`` scalar string dataset."""
        text = json.dumps(_make_json_safe(payload))
        parent.create_dataset("extras", data=np.bytes_(text.encode("utf-8")))

    def _write_vital(
        self, parent: h5py.Group, name: str, vit: Vital,
    ) -> None:
        v = parent.create_group(name)
        if name in _INT_VALUE_VITALS:
            value_data = np.int64(round(float(vit.value)))
        else:
            value_data = np.float64(vit.value)
        v.create_dataset("value", data=value_data)
        v.create_dataset("units", data=np.bytes_(vit.units.encode("utf-8")))
        v.create_dataset("timestamp", data=np.float64(vit.timestamp))
        v.create_dataset(
            "extras",
            data=np.bytes_(json.dumps(_make_json_safe(vit.extras)).encode("utf-8")),
        )

    @staticmethod
    def _set_attr(target, key: str, value: Any) -> None:
        """Coerce ``value`` into an h5py-friendly attribute representation."""
        if isinstance(value, str):
            target.attrs[key] = np.bytes_(value.encode("utf-8"))
        elif isinstance(value, bool):
            target.attrs[key] = np.bool_(value)
        elif isinstance(value, (int, np.integer)):
            target.attrs[key] = np.int64(value)
        elif isinstance(value, (float, np.floating)):
            target.attrs[key] = np.float64(value)
        elif isinstance(value, np.ndarray):
            target.attrs[key] = value
        else:
            target.attrs[key] = np.bytes_(str(value).encode("utf-8"))


# --------------------------------------------------------------------------- #
# JSON sanitisation
# --------------------------------------------------------------------------- #
def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
