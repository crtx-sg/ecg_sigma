"""Self-contained tests for the strict on-disk schema, including pacer
metadata, vital thresholds, and history extras."""

from __future__ import annotations

import json
import shutil
import tempfile

import h5py
import numpy as np
import pytest

from ecg_sigma.conditions import (
    AFIB,
    ALL_CONDITIONS,
    BRADYCARDIA,
    LBBB,
    NORMAL_SINUS,
    OTHER,
    PAC,
    PVC,
    TACHYCARDIA,
    VTACH,
    map_mitbih_rhythm,
    resolve_condition,
)
from ecg_sigma.loaders.base import BeatAnnotation, PatientRecord
from ecg_sigma.pipeline import Pipeline, PipelineConfig, event_uuid
from ecg_sigma.schema import (
    DEFAULT_LEADS,
    METHOD_DIRECT,
    METHOD_EINTHOVEN,
    METHOD_RULE_BASED,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
)
from ecg_sigma.signals.lead_mapper import LeadMapper
from ecg_sigma.signals.pacer import (
    PacerConfig,
    PacerGenerator,
    pack_pacer_info,
    unpack_pacer_info,
)
from ecg_sigma.signals.peaks import detect_r_peaks, heart_rate_bpm
from ecg_sigma.signals.processor import SignalProcessor
from ecg_sigma.signals.resampler import Resampler
from ecg_sigma.validation import (
    REQUIRED_METADATA_ATTRS,
    REQUIRED_VITALS,
    ValidationError,
    validate_pipeline_output,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _synth_ecg(fs: float, duration_s: float, hr_bpm: float = 75.0,
               seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros(n, dtype=np.float64)
    for i in range(int(duration_s / rr) + 2):
        center = (i + 0.5) * rr
        sig += 0.10 * np.exp(-((t - (center - 0.16)) / 0.025) ** 2)
        sig += -0.18 * np.exp(-((t - (center - 0.02)) / 0.012) ** 2)
        sig += 1.00 * np.exp(-((t - center) / 0.010) ** 2)
        sig += -0.20 * np.exp(-((t - (center + 0.02)) / 0.012) ** 2)
        sig += 0.25 * np.exp(-((t - (center + 0.30)) / 0.040) ** 2)
    sig += rng.normal(0.0, 0.01, size=n)
    return sig


def _synth_record(fs: float = 360.0, duration_s: float = 60.0,
                  hr_bpm: float = 75.0) -> PatientRecord:
    sig_mlii = _synth_ecg(fs, duration_s, hr_bpm=hr_bpm, seed=0)
    sig_v1 = 0.7 * np.roll(sig_mlii, int(0.02 * fs)) + np.random.default_rng(1).normal(
        0, 0.01, size=sig_mlii.size
    )
    rr = 60.0 / hr_bpm
    n = int(duration_s / rr)
    onsets = [int((i + 0.5) * rr * fs) for i in range(n)]
    beats = [BeatAnnotation(sample=s, symbol="N") for s in onsets]
    if len(beats) > 20:
        beats[20].symbol = "V"
    return PatientRecord(
        patient_id="SYN001",
        dataset="mitbih",
        fs=fs,
        signals={"MLII": sig_mlii, "V1": sig_v1},
        beat_annotations=beats,
        rhythm_annotations=[],
    )


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tobytes().decode("utf-8")
    return str(value)


def _read_json(ds) -> dict:
    return json.loads(_attr_str(np.array(ds)))


# --------------------------------------------------------------------------- #
# Resampler
# --------------------------------------------------------------------------- #
def test_resample_to_length_exact():
    x = np.random.default_rng(0).normal(size=3600).astype(np.float64)
    y = Resampler.resample_to_length(x, 360.0, 200.0, 2400)
    assert y.shape == (2400,)
    assert y.dtype == np.float32


def test_resample_irrational_ratio():
    x = np.random.default_rng(0).normal(size=2570).astype(np.float64)
    y = Resampler.resample_to_length(x, 257.0, 200.0, 2000)
    assert y.shape == (2000,)


# --------------------------------------------------------------------------- #
# R-peak detection
# --------------------------------------------------------------------------- #
def test_detect_r_peaks_recovers_rate():
    fs = 360.0
    ecg = _synth_ecg(fs, duration_s=20.0, hr_bpm=80.0)
    peaks = detect_r_peaks(ecg, fs)
    median_bpm, _ = heart_rate_bpm(peaks, fs)
    assert peaks.size >= 20
    assert abs(median_bpm - 80.0) < 5.0


def test_detect_r_peaks_empty_short():
    assert detect_r_peaks(np.zeros(10), fs=200.0).size == 0


# --------------------------------------------------------------------------- #
# Lead mapper
# --------------------------------------------------------------------------- #
def test_lead_mapper_einthoven_when_two_limb_leads():
    fs = 200.0
    n = int(fs * 8)
    rng = np.random.default_rng(42)
    I = rng.normal(0, 1, n).cumsum()
    II = rng.normal(0, 1, n).cumsum()
    mapped = LeadMapper().map({"I": I, "II": II}, fs=fs)
    assert mapped.leads["ECG3"].extras.method == METHOD_EINTHOVEN
    assert np.allclose(mapped.leads["ECG3"].signal,
                       (II - I).astype(np.float32), atol=1e-5)
    assert np.allclose(mapped.leads["aVR"].signal,
                       (-(I + II) / 2).astype(np.float32), atol=1e-5)


def test_lead_mapper_produces_seven_leads_from_two_lead_input():
    fs = 200.0
    ecg = _synth_ecg(fs, duration_s=12.0)
    v1 = 0.7 * np.roll(ecg, 4) + np.random.default_rng(0).normal(0, 0.005, size=ecg.size)
    mapped = LeadMapper().map({"MLII": ecg, "V1": v1}, fs=fs)
    assert set(mapped.leads.keys()) == set(DEFAULT_LEADS)
    assert mapped.leads["ECG2"].extras.source == SOURCE_REAL
    assert mapped.leads["ECG2"].extras.method == METHOD_DIRECT
    assert mapped.leads["ECG1"].extras.source == SOURCE_SYNTHETIC
    assert mapped.leads["ECG1"].extras.method == METHOD_RULE_BASED
    assert mapped.leads["vVX"].extras.source == SOURCE_REAL


# --------------------------------------------------------------------------- #
# Pacer generator
# --------------------------------------------------------------------------- #
def test_pacer_pack_unpack_round_trip():
    info = pack_pacer_info(2, 90, 7, 0)
    parts = unpack_pacer_info(info)
    assert parts == {"pacer_type": 2, "rate_bpm": 90, "amplitude": 7, "flags": 0}


def test_pacer_probabilities_match_spec():
    """VT/VF: ~40 %, Bradycardia: ~80 %, others: ~5 %."""
    rng = np.random.default_rng(0)
    gen = PacerGenerator(PacerConfig())
    n = 4000
    n_samples = 2400

    vt_on = sum(
        1 for _ in range(n)
        if gen.generate(VTACH, n_samples, rng)[0] != 0
    )
    brady_on = sum(
        1 for _ in range(n)
        if gen.generate(BRADYCARDIA, n_samples, rng)[0] != 0
    )
    norm_on = sum(
        1 for _ in range(n)
        if gen.generate(NORMAL_SINUS, n_samples, rng)[0] != 0
    )
    assert 0.34 < vt_on / n < 0.46, vt_on / n
    assert 0.74 < brady_on / n < 0.86, brady_on / n
    assert 0.02 < norm_on / n < 0.08, norm_on / n


def test_pacer_offset_within_window():
    rng = np.random.default_rng(1)
    gen = PacerGenerator(PacerConfig())
    for _ in range(500):
        info, off = gen.generate(VTACH, 2400, rng)
        assert 0 <= off < 2400
        if info != 0:
            assert off >= int(0.10 * 2400)


def test_pacer_no_pacer_means_zero_offset():
    """When pacer is off, both fields are zero."""
    rng = np.random.default_rng(0)
    cfg = PacerConfig(probabilities={}, default_probability=0.0)
    gen = PacerGenerator(cfg)
    for _ in range(100):
        info, off = gen.generate(NORMAL_SINUS, 2400, rng)
        assert info == 0 and off == 0


# --------------------------------------------------------------------------- #
# Strict on-disk schema
# --------------------------------------------------------------------------- #
def test_pipeline_output_matches_strict_schema():
    record = _synth_record()
    tmp = tempfile.mkdtemp(prefix="ecg_sigma_test_")
    try:
        cfg = PipelineConfig(output_dir=tmp, workers=1, random_seed=7)
        cfg.datasets["mitbih"] = {
            "enabled": False, "max_events_per_record": 5,
            "beat_symbols": ["N", "V"],
        }
        out_paths = Pipeline(cfg).process_record(record)
        assert out_paths

        for path in out_paths:
            warnings = validate_pipeline_output(path, cfg.schema, verify_history=True)
            assert isinstance(warnings, list)

            with h5py.File(path, "r") as f:
                md = f["metadata"].attrs
                for k in REQUIRED_METADATA_ATTRS:
                    assert k in md
                assert int(md["max_vital_history"]) == cfg.max_vital_history

                events = sorted(k for k in f.keys() if k.startswith("event_"))
                assert events[0] == "event_1001"
                evt = f[events[0]]

                # ECG extras = pacer-only.
                ecg_extras = _read_json(evt["ecg/extras"])
                assert set(ecg_extras.keys()) == {"pacer_info", "pacer_offset"}
                assert isinstance(ecg_extras["pacer_info"], int)
                assert isinstance(ecg_extras["pacer_offset"], int)

                # PPG / RESP extras = empty.
                assert _read_json(evt["ppg/extras"]) == {}
                assert _read_json(evt["resp/extras"]) == {}

                # Vitals: standard vitals carry thresholds + alarm_enabled + history.
                for name in REQUIRED_VITALS:
                    vg = evt[f"vitals/{name}"]
                    extras = _read_json(vg["extras"])
                    assert "history" in extras
                    assert isinstance(extras["history"], list)
                    if name == "XL_Posture":
                        assert "step_count" in extras
                        assert "time_since_posture_change" in extras
                    else:
                        assert extras["alarm_enabled"] is True
                        assert "upper_threshold" in extras
                        assert "lower_threshold" in extras

                # Temp must report degrees Fahrenheit.
                temp_units = _attr_str(np.array(evt["vitals/Temp/units"]))
                assert temp_units == "F"
                # RespRate units per spec.
                rr_units = _attr_str(np.array(evt["vitals/RespRate/units"]))
                assert rr_units == "breaths/min"

                # Lead provenance preserved as dataset attrs.
                ecg2 = evt["ecg/ECG2"]
                assert "source" in ecg2.attrs
                assert "method" in ecg2.attrs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_history_is_ascending_and_capped():
    record = _synth_record()
    tmp = tempfile.mkdtemp(prefix="ecg_sigma_test_")
    try:
        cfg = PipelineConfig(output_dir=tmp, workers=1, random_seed=11,
                             max_vital_history=20)
        cfg.datasets["mitbih"] = {
            "enabled": False, "max_events_per_record": 2,
            "beat_symbols": ["N", "V"],
        }
        paths = Pipeline(cfg).process_record(record)
        with h5py.File(paths[0], "r") as f:
            for ev in f:
                if not ev.startswith("event_"):
                    continue
                vg = f[ev]["vitals/HR"]
                hist = _read_json(vg["extras"])["history"]
                ts = [s["timestamp"] for s in hist]
                assert ts == sorted(ts)
                assert len(hist) <= cfg.max_vital_history
                # Most recent history sample is strictly before current.
                current_ts = float(np.array(vg["timestamp"]))
                assert hist[-1]["timestamp"] < current_ts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_history_values_stay_within_soft_range_at_extreme_hr():
    """Regression: at very high HR the trend baseline+jitter used to push
    history samples past the validator's soft range and fail the
    post-write check."""
    fs = 360.0
    duration = 30.0
    rr = 60.0 / 230.0                       # ~230 bpm, near the upper clip
    n = int(fs * duration)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for i in range(int(duration / rr) + 2):
        sig += 1.0 * np.exp(-((t - (i + 0.5) * rr) / 0.010) ** 2)
    sig += np.random.default_rng(0).normal(0, 0.005, n)

    beats = [
        BeatAnnotation(sample=int((i + 0.5) * rr * fs), symbol="V")
        for i in range(int(duration / rr))
    ]
    record = PatientRecord(
        patient_id="HR-EXTREME",
        dataset="synthetic",
        fs=fs,
        signals={"MLII": sig, "V1": np.roll(sig, 5)},
        beat_annotations=beats,
    )

    tmp = tempfile.mkdtemp(prefix="ecg_sigma_test_")
    try:
        cfg = PipelineConfig(output_dir=tmp, random_seed=99)
        cfg.datasets["synthetic"] = {
            "max_events_per_record": 4, "beat_symbols": ["V"],
        }
        paths = Pipeline(cfg).process_record(record)
        assert paths
        for p in paths:
            # verify_history=True is the default; calling it explicitly
            # so the regression intent is obvious in the test name.
            validate_pipeline_output(p, verify_history=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pipeline_is_deterministic():
    record = _synth_record()
    cfg = PipelineConfig(workers=1, random_seed=11)
    cfg.datasets["mitbih"] = {
        "enabled": False, "max_events_per_record": 3, "beat_symbols": ["N", "V"],
    }
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        cfg_a = PipelineConfig(**{**cfg.__dict__, "output_dir": a})
        cfg_b = PipelineConfig(**{**cfg.__dict__, "output_dir": b})
        out_a = Pipeline(cfg_a).process_record(record)
        out_b = Pipeline(cfg_b).process_record(record)
        with h5py.File(out_a[0], "r") as fa, h5py.File(out_b[0], "r") as fb:
            assert np.array_equal(
                fa["event_1001/ecg/ECG2"][...],
                fb["event_1001/ecg/ECG2"][...],
            )
            ts_a = float(np.array(fa["event_1001/timestamp"]))
            ts_b = float(np.array(fb["event_1001/timestamp"]))
            assert ts_a == ts_b
            # Pacer is also deterministic.
            assert (_read_json(fa["event_1001/ecg/extras"])
                    == _read_json(fb["event_1001/ecg/extras"]))


# --------------------------------------------------------------------------- #
# Label mapping and reconciliation
# --------------------------------------------------------------------------- #
def test_rhythm_aux_note_tolerates_wfdb_nul_padding():
    """WFDB pads aux notes to an even byte count with a trailing NUL.

    Without an explicit strip, 94% of MIT-BIH rhythm annotations parse as
    None and VTACH/BRADYCARDIA/AFIB vanish from the output entirely.
    """
    for raw, expected in (
        ("(AFIB\x00", AFIB),
        ("(AFIB", AFIB),
        (" (VT\x00 ", VTACH),
        ("(SBR\x00", BRADYCARDIA),
    ):
        assert map_mitbih_rhythm(raw) == expected, raw
    assert map_mitbih_rhythm("") is None
    assert map_mitbih_rhythm("\x00") is None


def test_wfdb_bigeminy_trigeminy_are_not_rate_alarms():
    """`(B` is ventricular BIGEMINY and `(T` is TRIGEMINY.

    Reading them as bradycardia/tachycardia invents ~1400 phantom
    rate-alarm events across MIT-BIH.
    """
    assert map_mitbih_rhythm("(B\x00") == PVC
    assert map_mitbih_rhythm("(T\x00") == PVC
    assert map_mitbih_rhythm("(SBR\x00") == BRADYCARDIA
    assert map_mitbih_rhythm("(SVTA\x00") == TACHYCARDIA


def test_resolve_condition_keeps_ectopics_and_promotes_runs():
    # A PVC inside a sinus strip stays a PVC ...
    assert resolve_condition(PVC, NORMAL_SINUS) == PVC
    # ... but a V beat inside a VT run is announced as VTACH.
    assert resolve_condition(PVC, VTACH) == VTACH
    # Rate alarms outrank beat morphology.
    assert resolve_condition(PAC, BRADYCARDIA) == BRADYCARDIA
    # Rhythm-only and beat-only both pass through.
    assert resolve_condition(None, AFIB) == AFIB
    assert resolve_condition(LBBB, None) == LBBB
    assert resolve_condition(None, None) == OTHER


# --------------------------------------------------------------------------- #
# Lead mapping preconditions
# --------------------------------------------------------------------------- #
def test_lead_mapper_refuses_precordial_only_input():
    """MIT-BIH 102/104 carry V5+V2 and no limb lead.

    Six of seven output leads plus HR/PPG/RESP descend from Lead II, so a
    zero-filled montage would be pure invention that still passes a
    structural check.
    """
    n = 2400
    v5 = np.sin(np.linspace(0, 40, n))
    v2 = np.cos(np.linspace(0, 40, n))
    with pytest.raises(ValueError, match="no limb leads"):
        LeadMapper().map({"V5": v5, "V2": v2}, fs=200.0)


def test_pipeline_raises_configuration_error_on_short_record():
    """PTB-XL's 10 s records against the default 12 s window."""
    from ecg_sigma.pipeline import ConfigurationError

    fs = 500.0
    n = int(10 * fs)
    sig = _synth_ecg(fs, 10.0, hr_bpm=70.0)
    record = PatientRecord(
        patient_id="SHORT001", dataset="ptbxl", fs=fs,
        signals={"I": sig, "II": sig * 1.1},
        beat_annotations=[BeatAnnotation(sample=n // 2, symbol="N")],
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(output_dir=tmp, workers=1)
        with pytest.raises(ConfigurationError, match="schema window is 12s"):
            Pipeline(cfg).process_record(record)


# --------------------------------------------------------------------------- #
# Content validation
# --------------------------------------------------------------------------- #
def _write_one_file(tmp: str, seed: int = 3) -> str:
    cfg = PipelineConfig(output_dir=tmp, workers=1, random_seed=seed)
    cfg.datasets["mitbih"] = {
        "enabled": False, "max_events_per_record": 2, "beat_symbols": ["N", "V"],
    }
    paths = Pipeline(cfg).process_record(_synth_record())
    assert paths
    return paths[0]


def test_validator_rejects_flat_lead():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r+") as f:
            f["event_1001/ecg/ECG1"][...] = 0.0
        with pytest.raises(ValidationError, match="flat"):
            validate_pipeline_output(path)


def test_validator_rejects_condition_outside_vocabulary():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r+") as f:
            f["event_1001"].attrs["condition"] = np.bytes_(b"SOMETHING_ELSE")
        with pytest.raises(ValidationError, match="not in the unified"):
            validate_pipeline_output(path)


def test_validator_requires_lead_provenance_attrs():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r+") as f:
            del f["event_1001/ecg/ECG2"].attrs["source"]
        with pytest.raises(ValidationError, match="missing provenance attr"):
            validate_pipeline_output(path)


def test_validator_detects_heart_rate_disagreeing_with_vitals():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r+") as f:
            f["event_1001"].attrs["heart_rate"] = np.float64(180.0)
        with pytest.raises(ValidationError, match="disagrees with"):
            validate_pipeline_output(path)


def test_validator_warns_on_low_quality_score():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r+") as f:
            f["metadata"].attrs["data_quality_score"] = np.float64(0.05)
        warnings = validate_pipeline_output(path)
        assert any("data_quality_score" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Quality score
# --------------------------------------------------------------------------- #
def test_quality_score_ranks_clean_above_noisy_and_flat():
    fs = 200.0
    clean = _synth_ecg(fs, 12.0, hr_bpm=70.0, seed=0)
    rng = np.random.default_rng(5)
    noisy = clean + rng.normal(0.0, 0.6, size=clean.size)
    flat = np.zeros_like(clean)

    q_clean = SignalProcessor.quality_score(clean, fs)
    q_noisy = SignalProcessor.quality_score(noisy, fs)
    q_flat = SignalProcessor.quality_score(flat, fs)

    assert q_flat == 0.0
    assert q_noisy < q_clean
    # The old metric measured power above 40 Hz -- the bandpass stopband --
    # and so pinned everything at ~1.0. Guard against regressing to that.
    assert q_clean < 0.99


# --------------------------------------------------------------------------- #
# Determinism and traceability
# --------------------------------------------------------------------------- #
def test_pipeline_output_is_byte_identical_across_runs():
    """The whole file, not just the signal arrays.

    Event UUIDs used to come from uuid4, so two runs never matched.
    """
    record = _synth_record()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        path_a = _write_one_file(a, seed=13)
        path_b = _write_one_file(b, seed=13)
        assert open(path_a, "rb").read() == open(path_b, "rb").read()


def test_event_uuid_is_stable_under_event_filtering():
    """Keyed on the source sample, not the output index."""
    first = event_uuid("mitbih", "208", 32719)
    assert first == event_uuid("mitbih", "208", 32719)
    assert first != event_uuid("mitbih", "208", 32720)
    assert first != event_uuid("incart", "208", 32719)


def test_output_carries_source_traceability():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_one_file(tmp)
        with h5py.File(path, "r") as f:
            md = f["metadata"].attrs
            assert _attr_str(md["source_dataset"]) == "mitbih"
            assert _attr_str(md["source_channels"]) == "MLII,V1"
            assert float(md["source_sampling_rate"]) == 360.0

            evt = f["event_1001"]
            assert _attr_str(evt.attrs["source_label"]) in ("N", "V")
            assert int(evt.attrs["source_sample"]) >= 0
            # Both label inputs survive, so a consumer can re-derive a
            # morphology-first label without re-running the pipeline.
            beat = _attr_str(evt.attrs["source_beat_condition"])
            assert beat in ALL_CONDITIONS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
