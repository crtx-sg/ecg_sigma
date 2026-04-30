"""Produce a sample HDF5 file using a synthetic ECG record.

Useful for inspecting the schema without downloading any dataset:

    python -m scripts.generate_sample --out ./samples
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from ecg_sigma.loaders.base import BeatAnnotation, PatientRecord  # noqa: E402
from ecg_sigma.pipeline import Pipeline, PipelineConfig  # noqa: E402


def _synth_ecg(fs: float, duration_s: float, hr_bpm: float, seed: int) -> np.ndarray:
    """Synthetic ECG: sum of P/Q/R/S/T Gaussians, jittered for naturalism."""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros(n, dtype=np.float64)
    for i in range(int(duration_s / rr) + 2):
        center = (i + 0.5) * rr + rng.normal(0, 0.01)
        sig += 0.10 * np.exp(-((t - (center - 0.16)) / 0.025) ** 2)
        sig += -0.18 * np.exp(-((t - (center - 0.02)) / 0.012) ** 2)
        sig += 1.00 * np.exp(-((t - center) / 0.010) ** 2)
        sig += -0.20 * np.exp(-((t - (center + 0.02)) / 0.012) ** 2)
        sig += 0.25 * np.exp(-((t - (center + 0.30)) / 0.040) ** 2)
    sig += rng.normal(0.0, 0.01, size=n)
    return sig


def make_record(duration_s: float = 120.0, hr_bpm: float = 72.0,
                fs: float = 360.0) -> PatientRecord:
    mlii = _synth_ecg(fs, duration_s, hr_bpm=hr_bpm, seed=0)
    v1 = (
        0.7 * np.roll(mlii, int(0.02 * fs))
        + np.random.default_rng(1).normal(0, 0.01, size=mlii.size)
    )
    rr = 60.0 / hr_bpm
    n_beats = int(duration_s / rr)
    beats = [
        BeatAnnotation(sample=int((i + 0.5) * rr * fs), symbol="N")
        for i in range(n_beats)
    ]
    # Sprinkle in a couple of PVCs and one APC for label diversity.
    for idx, sym in [(int(n_beats * 0.30), "V"),
                     (int(n_beats * 0.55), "A"),
                     (int(n_beats * 0.78), "V")]:
        if 0 <= idx < len(beats):
            beats[idx].symbol = sym
    return PatientRecord(
        patient_id="SYN-DEMO-0001",
        dataset="synthetic",
        fs=fs,
        signals={"MLII": mlii, "V1": v1},
        beat_annotations=beats,
        rhythm_annotations=[],
        metadata={"description": "Synthetic record from generate_sample.py"},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit a sample HDF5 file")
    p.add_argument("--out", default="./samples", help="output dir")
    p.add_argument("--max-events", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        output_dir=args.out,
        random_seed=args.seed,
        workers=1,
        datasets={
            "synthetic": {
                "enabled": False,                 # we drive process_record directly
                "max_events_per_record": args.max_events,
                "beat_symbols": ["N", "V", "A"],
            }
        },
    )
    paths = Pipeline(cfg).process_record(make_record())
    for path in paths:
        print(path)
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
