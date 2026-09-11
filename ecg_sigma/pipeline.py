"""End-to-end pipeline: dataset records -> per-patient HDF5 files.

Flow per record:

    DatasetLoader.iter_records()
        -> EventExtractor.extract()
            -> for each event:
                * crop ECG window in source-fs
                * SignalProcessor.preprocess() per channel
                * Resampler -> 200 Hz, exact length
                * LeadMapper -> 7-lead canonical
                * ModalitiesSynthesizer -> PPG, RESP
                * VitalsGenerator -> dict[name] = Vital
                * SignalProcessor.quality_score()
        -> bucket by (patient_id, year, month)
        -> HDF5Writer.write() per bucket
        -> validate_pipeline_output() (smoke check)

The pipeline is single-record-at-a-time so it scales to arbitrary dataset
sizes: loaders are generators and :meth:`Pipeline.run` never materialises
them. Optional record-level parallelism is exposed via the ``workers``
config field, bounded to ``2 * workers`` resident records.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .conditions import OTHER, NORMAL_SINUS
from .events import (
    BeatBasedExtractor,
    Event,
    EventExtractor,
    RhythmBasedExtractor,
)
from .events.beat_based import BeatExtractorConfig
from .events.rhythm_based import RhythmExtractorConfig
from .loaders import DatasetLoader, INCARTLoader, MITBIHLoader, PTBXLLoader
from .loaders.base import PatientRecord
from .schema import SCHEMA, SchemaSpec, ExtrasTag
from .signals import (
    LeadMapper,
    ModalitiesSynthesizer,
    PacerGenerator,
    Resampler,
    SignalProcessor,
)
from .signals.pacer import PacerConfig
from .signals.peaks import detect_r_peaks, heart_rate_bpm
from .signals.processor import ProcessorConfig
from .signals.synthesis import SynthesisConfig
from .utils import configure_logging, get_logger, seeded_rng
from .validation import ValidationError, validate_event_payload, validate_pipeline_output
from .vitals import Vital, VitalsGenerator
from .vitals.generator import VitalsConfig
from .writers import HDF5Writer
from .writers.hdf5_writer import EventPayload, FilePayload

_log = get_logger(__name__)

# Namespace for deterministic event UUIDs. Fixed for the life of the schema:
# changing it renames every event in every previously-generated file.
_EVENT_UUID_NAMESPACE = uuid.UUID("6f9d1c2e-0b47-5a3e-9c81-2f5b7d4a6e30")


class ConfigurationError(Exception):
    """The config and the dataset cannot produce output as combined.

    Distinct from a per-record data error: skipping and retrying will not
    help, so :meth:`Pipeline.run` re-raises this even when ``fail_fast``
    is off rather than logging it once per record.
    """


def event_uuid(dataset: str, patient_id: str, source_sample: int) -> str:
    """Deterministic per-event UUID.

    Keyed on the event's *identity* in the source data rather than on its
    index in the output file, so filtering or re-ordering events does not
    renumber the survivors, and two runs of the same input agree
    byte-for-byte.
    """
    return str(uuid.uuid5(
        _EVENT_UUID_NAMESPACE, f"{dataset}/{patient_id}/{source_sample}"
    ))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Resolved pipeline configuration. Build via :func:`load_config`."""

    output_dir: str = "./out"
    file_pattern: str = "{patient_id}_{year}-{month:02d}.h5"
    compression: str = "gzip"
    compression_opts: int = 4
    timestamp_anchor_iso: str = "2025-01-01T00:00:00Z"

    schema: SchemaSpec = SCHEMA
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    vitals: VitalsConfig = field(default_factory=VitalsConfig)
    pacer: PacerConfig = field(default_factory=PacerConfig)

    rpeak_min_distance_s: float = 0.30
    rpeak_min_height_z: float = 0.6

    random_seed: int = 42
    log_level: str = "INFO"
    workers: int = 1
    fail_fast: bool = False
    max_vital_history: int = 30

    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load and validate pipeline config from a YAML file (or dict)."""
    cfg_data = _load_default_yaml()
    if path:
        cfg_data = _deep_update(cfg_data, _load_yaml(path))
    if overrides:
        cfg_data = _deep_update(cfg_data, overrides)
    return _build_config(cfg_data)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """High-level pipeline. Construct once, call :meth:`run`."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        configure_logging(cfg.log_level)
        self.processor = SignalProcessor(cfg.processor)
        self.lead_mapper = LeadMapper()
        self.vitals_gen = VitalsGenerator(cfg.vitals)
        self.pacer_gen = PacerGenerator(cfg.pacer)
        self.writer = HDF5Writer(
            output_dir=cfg.output_dir,
            schema=cfg.schema,
            compression=cfg.compression,
            compression_opts=cfg.compression_opts,
            file_pattern=cfg.file_pattern,
        )

    # ------------------------------------------------------------------ #
    # Top-level API
    # ------------------------------------------------------------------ #
    def run(self) -> List[str]:
        """Process all enabled datasets. Returns the list of written paths.

        Records are *streamed* from the loaders, never materialised into a
        list: one INCART record is ~44 MB of float64 signal (12 leads x 30
        min @ 257 Hz), so holding all 75 costs ~3.3 GB before any work
        starts, and PTB-XL's 21,837 records would be hopeless.
        """
        if self.cfg.workers <= 1:
            outputs: List[str] = []
            seen = 0
            for rec in self._collect_records():
                seen += 1
                try:
                    outputs.extend(self.process_record(rec))
                except ConfigurationError:
                    raise
                except Exception:
                    if self.cfg.fail_fast:
                        raise
                    _log.exception("failed to process record %s", rec.patient_id)
            if not seen:
                _log.warning("no records to process")
            return outputs
        return self._run_parallel(self._collect_records())

    def process_record(self, record: PatientRecord) -> List[str]:
        """Process one :class:`PatientRecord` end-to-end. Returns written paths."""
        self._check_record_length(record)
        rng = seeded_rng(self.cfg.random_seed, record.patient_id)
        extractor = self._build_extractor(record)
        events = extractor.extract(record)
        _log.info(
            "record=%s dataset=%s extracted_events=%d",
            record.patient_id, record.dataset, len(events),
        )
        if not events:
            return []

        # Build an EventPayload per Event.
        payloads: List[EventPayload] = []
        for evt in events:
            pl = self._build_event_payload(record, evt, rng)
            if pl is None:
                continue
            try:
                validate_event_payload(pl, self.cfg.schema)
            except ValidationError as exc:
                _log.warning("dropping invalid event for %s: %s", record.patient_id, exc)
                continue
            payloads.append(pl)

        if not payloads:
            _log.warning("record=%s produced 0 valid events", record.patient_id)
            return []

        # Bucket by (year, month). For datasets with no real time, all
        # events end up in the anchor month.
        buckets: Dict[Tuple[int, int], List[EventPayload]] = defaultdict(list)
        for pl in payloads:
            ts = float(pl.event_timestamp_epoch)
            d = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
            buckets[(d.year, d.month)].append(pl)

        out_paths: List[str] = []
        for (yr, mo), evts in sorted(buckets.items()):
            file_payload = FilePayload(
                patient_id=record.patient_id,
                dataset=record.dataset,
                year=yr,
                month=mo,
                events=evts,
                record_metadata=record.metadata,
                max_vital_history=self.cfg.max_vital_history,
                source_fs=float(record.fs),
                source_channels=tuple(record.signals.keys()),
            )
            path = self.writer.write(file_payload)
            warnings = validate_pipeline_output(path, self.cfg.schema)
            for w in warnings:
                _log.warning("post-write: %s :: %s", os.path.basename(path), w)
            out_paths.append(path)
        return out_paths

    # ------------------------------------------------------------------ #
    # Per-event construction
    # ------------------------------------------------------------------ #
    def _build_event_payload(
        self,
        record: PatientRecord,
        evt: Event,
        rng: np.random.Generator,
    ) -> Optional[EventPayload]:
        schema = self.cfg.schema
        fs_in = float(record.fs)

        # 1. Crop window in source-fs coordinates.
        before = int(round(schema.seconds_before_event * fs_in))
        after = int(round(schema.seconds_after_event * fs_in))
        first = next(iter(record.signals.values()))
        n_total = first.size
        lo = evt.onset_sample - before
        hi = evt.onset_sample + after
        if lo < 0 or hi > n_total:
            _log.debug(
                "event @%d outside window [%d,%d] for record %s; skipping",
                evt.onset_sample, lo, hi, record.patient_id,
            )
            return None

        # 2. Pre-process each available channel + resample to ECG fs.
        n_out = schema.ecg_samples
        resampled: Dict[str, np.ndarray] = {}
        for name, sig in record.signals.items():
            window = np.asarray(sig[lo:hi], dtype=np.float64)
            cleaned = self.processor.preprocess(window, fs_in)
            resampled[name] = Resampler.resample_to_length(
                cleaned, fs_in, schema.ecg_fs, n_out
            )

        # 3. Map heterogeneous leads to canonical 7-lead set.
        mapped = self.lead_mapper.map(resampled, fs=schema.ecg_fs)

        # 4. R-peaks on Lead II for HR + downstream synthesis.
        ecg_ii = mapped.leads["ECG2"].signal.astype(np.float64)
        peaks = detect_r_peaks(
            ecg_ii, schema.ecg_fs,
            min_distance_s=self.cfg.rpeak_min_distance_s,
            min_height_z=self.cfg.rpeak_min_height_z,
        )
        hr_med, _ = heart_rate_bpm(peaks, schema.ecg_fs)

        # 5. Synthesise PPG + RESP.
        synth = ModalitiesSynthesizer(schema, self.cfg.synthesis, rng)
        modalities = synth.synthesize(ecg_ii)

        # 6. Vitals (RR rate from RESP via FFT peak; falls back to HR-based).
        rr_rate_brpm = self._estimate_resp_rate(modalities.resp, schema.resp_fs)
        evt_ts = self._event_timestamp(record, evt)
        vitals = self.vitals_gen.generate(
            condition=evt.condition,
            hr_bpm=hr_med,
            resp_rate_brpm=rr_rate_brpm,
            event_timestamp_epoch=evt_ts,
            rng=rng,
            max_vital_history=self.cfg.max_vital_history,
        )

        # 6b. Pacer metadata for ECG extras.
        pacer_info, pacer_offset = self.pacer_gen.generate(
            evt.condition, schema.ecg_samples, rng,
        )

        # 7. Quality score on the canonical Lead II.
        quality = self.processor.quality_score(ecg_ii, schema.ecg_fs)

        # 8. Build the writer payload.
        ecg_payload = {
            lead: (mapped.leads[lead].signal, mapped.leads[lead].extras)
            for lead in schema.leads
        }
        return EventPayload(
            condition=evt.condition,
            source_label=evt.source_label,
            event_timestamp_epoch=evt_ts,
            heart_rate_bpm=float(hr_med),
            data_quality_score=float(quality),
            ecg=ecg_payload,
            ppg=(modalities.ppg, modalities.ppg_extras),
            resp=(modalities.resp, modalities.resp_extras),
            vitals=vitals,
            pacer_info=int(pacer_info),
            pacer_offset=int(pacer_offset),
            extras={"uuid": event_uuid(
                record.dataset, record.patient_id, evt.onset_sample,
            )},
            source_sample=int(evt.onset_sample),
            source_beat_condition=str(evt.metadata.get("beat_condition", "")),
            source_rhythm_condition=str(evt.metadata.get("rhythm_context", "")),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _check_record_length(self, record: PatientRecord) -> None:
        """Refuse a record shorter than one output window.

        Every event needs ``seconds_before + seconds_after`` of real signal.
        A record shorter than that yields nothing at all, and the per-event
        drop is easy to miss -- PTB-XL (10 s records against the default
        12 s window) silently produced zero output for all 21,837 records.
        Fail once, with the arithmetic and the fix.
        """
        schema = self.cfg.schema
        if not record.signals:
            raise ConfigurationError(
                f"record {record.patient_id} ({record.dataset}) has no signals"
            )
        n_total = min(int(getattr(s, "size", 0)) for s in record.signals.values())
        needed = int(round(schema.window_seconds * float(record.fs)))
        if n_total >= needed:
            return
        max_half = int(n_total / float(record.fs)) // 2
        raise ConfigurationError(
            f"dataset {record.dataset!r}: record {record.patient_id} is "
            f"{n_total / float(record.fs):.1f}s at {record.fs:g} Hz, but the "
            f"schema window is {schema.window_seconds}s "
            f"(seconds_before_event={schema.seconds_before_event} + "
            f"seconds_after_event={schema.seconds_after_event}). Every event "
            f"would fall off the record, so this dataset cannot be converted "
            f"at the current window. Either set seconds_before_event and "
            f"seconds_after_event to at most {max_half} each -- which produces "
            f"a {int(2 * max_half * schema.ecg_fs)}-sample variant that is NOT "
            f"interchangeable with the {schema.ecg_samples}-sample output from "
            f"longer datasets -- or disable {record.dataset!r}."
        )

    def _build_extractor(self, record: PatientRecord) -> EventExtractor:
        """Pick beat- vs rhythm-based extractor by what the record carries."""
        ds_cfg = self.cfg.datasets.get(record.dataset, {}) or {}
        if record.beat_annotations:
            # Reject beats whose 12-second window would fall off the record.
            margin = int(round(
                self.cfg.schema.seconds_before_event * float(record.fs)
            ))
            return BeatBasedExtractor(BeatExtractorConfig(
                include_symbols=tuple(ds_cfg.get("beat_symbols") or ()) or None,
                max_events_per_record=ds_cfg.get("max_events_per_record"),
                stride=int(ds_cfg.get("stride", 1)),
                drop_other=bool(ds_cfg.get("drop_other", False)),
                rng_seed=self.cfg.random_seed,
                safety_margin_samples=margin,
            ))
        return RhythmBasedExtractor(RhythmExtractorConfig(
            n_events=int(ds_cfg.get("max_events_per_record", 1) or 1),
            label_set="ptbxl",
        ))

    def _event_timestamp(self, record: PatientRecord, evt: Event) -> float:
        """Compute a deterministic epoch-second timestamp (float) for an event.

        Datasets without absolute time (MIT-BIH, INCART) get anchored to
        ``timestamp_anchor_iso`` plus the per-record offset. The result is
        identical across runs and across worker counts for the same input.
        """
        sub_second_offset = evt.onset_sample / float(record.fs)
        if record.base_time_epoch:
            return float(record.base_time_epoch) + sub_second_offset
        anchor = _parse_iso_z(self.cfg.timestamp_anchor_iso)
        # Spread records across days deterministically via blake2b on the id
        # (Python's built-in hash is salted per-process, so it is unsafe here).
        import hashlib
        digest = hashlib.blake2b(record.patient_id.encode("utf-8"), digest_size=8).digest()
        per_record_offset = int.from_bytes(digest, "big") % (60 * 60 * 24 * 28)
        return float(anchor.timestamp() + per_record_offset + sub_second_offset)

    def _estimate_resp_rate(self, resp: np.ndarray, fs: float) -> float:
        """Crude FFT-based respiration rate (in breaths per minute)."""
        if resp.size < 8:
            return 15.0
        x = resp - resp.mean()
        n = x.size
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        # Look between 0.1 and 0.8 Hz (6-48 brpm).
        mask = (freqs >= 0.1) & (freqs <= 0.8)
        if not mask.any():
            return 15.0
        peak = freqs[mask][int(np.argmax(np.abs(spec[mask])))]
        return float(peak * 60.0)

    # ------------------------------------------------------------------ #
    # Dataset enumeration
    # ------------------------------------------------------------------ #
    def _collect_records(self) -> Iterable[PatientRecord]:
        for dataset_name, ds_cfg in self.cfg.datasets.items():
            if not ds_cfg or not ds_cfg.get("enabled", False):
                continue
            loader = self._build_loader(dataset_name, ds_cfg)
            if loader is None:
                continue
            yield from loader.iter_records()

    @staticmethod
    def _build_loader(name: str, cfg: Dict[str, Any]) -> Optional[DatasetLoader]:
        path = cfg.get("path")
        if not path:
            _log.warning("dataset %s enabled but no path; skipping", name)
            return None
        if name == "mitbih":
            return MITBIHLoader(path, record_pattern=cfg.get("record_pattern"))
        if name == "incart":
            return INCARTLoader(path, record_pattern=cfg.get("record_pattern"))
        if name == "ptbxl":
            return PTBXLLoader(
                path,
                sampling_rate=int(cfg.get("sampling_rate", 500)),
                max_records=cfg.get("max_records"),
            )
        _log.warning("unknown dataset %s", name)
        return None

    # ------------------------------------------------------------------ #
    # Multi-process
    # ------------------------------------------------------------------ #
    def _run_parallel(self, records: Iterable[PatientRecord]) -> List[str]:
        """Fan out over a bounded window of in-flight records.

        Submitting every record up-front would pull the whole dataset into
        the parent process (and again into the pickle buffers). We keep at
        most ``2 * workers`` records resident, pulling the next one from the
        loader only as a slot frees up.
        """
        outputs: List[str] = []
        # The Pipeline holds simple, picklable config; workers just call
        # process_record on a freshly-constructed Pipeline.
        cfg = self.cfg
        max_inflight = max(2, cfg.workers * 2)
        source = iter(records)
        seen = 0

        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures: Dict[Any, str] = {}

            def submit_next() -> bool:
                nonlocal seen
                try:
                    rec = next(source)
                except StopIteration:
                    return False
                futures[pool.submit(_worker_process_record, cfg, rec)] = rec.patient_id
                seen += 1
                return True

            for _ in range(max_inflight):
                if not submit_next():
                    break

            while futures:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    patient_id = futures.pop(fut)
                    try:
                        outputs.extend(fut.result())
                    except ConfigurationError:
                        raise
                    except Exception:
                        if cfg.fail_fast:
                            raise
                        _log.exception("worker failed on record %s", patient_id)
                    submit_next()

        if not seen:
            _log.warning("no records to process")
        return outputs


def _worker_process_record(cfg: PipelineConfig, record: PatientRecord) -> List[str]:
    """Top-level helper so ``ProcessPoolExecutor`` can pickle it."""
    return Pipeline(cfg).process_record(record)


# --------------------------------------------------------------------------- #
# Config-loading helpers
# --------------------------------------------------------------------------- #
def _load_default_yaml() -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.normpath(os.path.join(here, "..", "config", "default.yaml"))
    if not os.path.exists(default_path):
        return {}
    return _load_yaml(default_path)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # PyYAML
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML config") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _build_config(data: Dict[str, Any]) -> PipelineConfig:
    schema_data = data.get("schema") or {}
    schema = SchemaSpec(
        ecg_fs=float(schema_data.get("ecg_fs", 200.0)),
        ppg_fs=float(schema_data.get("ppg_fs", 75.0)),
        resp_fs=float(schema_data.get("resp_fs", 100.0 / 3.0)),
        seconds_before_event=int(schema_data.get("seconds_before_event", 6)),
        seconds_after_event=int(schema_data.get("seconds_after_event", 6)),
        device_info=str(schema_data.get("device_info", "RMSAI-SimDevice-v2.0")),
        leads=tuple(schema_data.get("leads", SCHEMA.leads)),
    )

    sig = data.get("signal") or {}
    proc = ProcessorConfig(
        bandpass_hz=tuple(sig.get("ecg_bandpass_hz", (0.5, 40.0))),
        notch_hz=sig.get("ecg_notch_hz", 50.0),
        notch_q=float(sig.get("ecg_notch_q", 30.0)),
    )
    syn = data.get("synthesis") or {}
    synthesis = SynthesisConfig(
        ppg_pulse_delay_s=float(syn.get("ppg_pulse_delay_s", 0.20)),
        ppg_systolic_width_s=float(syn.get("ppg_systolic_width_s", 0.10)),
        ppg_dicrotic_offset_s=float(syn.get("ppg_dicrotic_offset_s", 0.30)),
        ppg_dicrotic_amp=float(syn.get("ppg_dicrotic_amp", 0.35)),
        resp_method=str(syn.get("resp_method", "edr")),
        resp_default_rate_brpm=float(syn.get("resp_default_rate_brpm", 15.0)),
        resp_noise_std=float(syn.get("resp_noise_std", 0.05)),
    )
    vit = data.get("vitals") or {}
    vitals_cfg = VitalsConfig(
        spo2={k: tuple(v) for k, v in (vit.get("spo2") or {}).items()},
        bp_systolic={k: tuple(v) for k, v in (vit.get("bp_systolic") or {}).items()},
        bp_diastolic_offset=tuple(vit.get("bp_diastolic_offset", (55.0, 85.0))),
        temp_f=tuple(vit.get("temp_f", (97.5, 99.5))),
        postures=tuple(vit.get("postures", ("Supine", "LeftLat", "RightLat",
                                            "Prone", "Sitting", "Standing"))),
        thresholds={k: tuple(v) for k, v in (vit.get("thresholds") or {}).items()},
        step_count_range=tuple(vit.get("step_count_range", (0, 3000))),
        time_since_posture_change_range=tuple(
            vit.get("time_since_posture_change_range", (0, 3600))
        ),
    )

    pacer_data = data.get("pacer") or {}
    pacer_cfg = PacerConfig(
        probabilities={k: float(v) for k, v in (pacer_data.get("probabilities") or {}).items()
                       } or PacerConfig().probabilities,
        default_probability=float(pacer_data.get("default_probability", 0.05)),
    )
    rt = data.get("runtime") or {}
    return PipelineConfig(
        output_dir=str(data.get("output_dir", "./out")),
        file_pattern=str(data.get("file_pattern", "{patient_id}_{year}-{month:02d}.h5")),
        compression=str(data.get("compression", "gzip")),
        compression_opts=int(data.get("compression_opts", 4)),
        timestamp_anchor_iso=str(data.get("timestamp_anchor", "2025-01-01T00:00:00Z")),
        schema=schema,
        processor=proc,
        synthesis=synthesis,
        vitals=vitals_cfg,
        pacer=pacer_cfg,
        rpeak_min_distance_s=float(sig.get("rpeak_min_distance_s", 0.30)),
        rpeak_min_height_z=float(sig.get("rpeak_min_height_z", 0.6)),
        random_seed=int(rt.get("random_seed", 42)),
        log_level=str(rt.get("log_level", "INFO")),
        workers=int(rt.get("workers", 1)),
        fail_fast=bool(rt.get("fail_fast", False)),
        max_vital_history=int(data.get("max_vital_history", 30)),
        datasets=dict(data.get("datasets") or {}),
    )


def _parse_iso_z(s: str) -> _dt.datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return _dt.datetime.fromisoformat(s)
