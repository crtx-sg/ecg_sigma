"""Self-contained tests for the strict on-disk schema, including pacer
metadata, vital thresholds, and history extras."""

from __future__ import annotations

import json
import shutil
import tempfile

import h5py
import numpy as np
import pytest

from ecg_sigma.conditions import BRADYCARDIA, NORMAL_SINUS, VTACH
from ecg_sigma.loaders.base import BeatAnnotation, PatientRecord
from ecg_sigma.pipeline import Pipeline, PipelineConfig
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
from ecg_sigma.signals.resampler import Resampler
from ecg_sigma.validation import (
    REQUIRED_METADATA_ATTRS,
    REQUIRED_VITALS,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
