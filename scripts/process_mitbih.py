"""End-to-end example: process the MIT-BIH Arrhythmia Database.

Usage::

    # Default config (config/default.yaml), data path overridden on CLI:
    python -m scripts.process_mitbih --mitbih-path /path/to/mit-bih --out ./out

    # Custom config:
    python -m scripts.process_mitbih --config my.yaml

    # Restrict to a subset of records (handy for quick smoke tests):
    python -m scripts.process_mitbih --mitbih-path /path/to/mit-bih \\
        --records 100 101 --max-events-per-record 20

The script does not auto-download the dataset; download it from PhysioNet
(https://physionet.org/content/mitdb/1.0.0/) and pass the directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

# Allow running the script directly from the repo root via `python scripts/...`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecg_sigma.pipeline import Pipeline, load_config  # noqa: E402
from ecg_sigma.utils import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process MIT-BIH -> ICU HDF5")
    p.add_argument("--config", help="Optional YAML overriding config/default.yaml")
    p.add_argument("--mitbih-path", help="Directory containing MIT-BIH .hea/.dat/.atr")
    p.add_argument("--out", help="Output directory (default: from config)")
    p.add_argument(
        "--records", nargs="*",
        help="Restrict to specific record IDs (e.g. --records 100 101)",
    )
    p.add_argument(
        "--max-events-per-record", type=int,
        help="Cap events per record (overrides config)",
    )
    p.add_argument(
        "--workers", type=int, help="Number of parallel record workers",
    )
    p.add_argument("--seed", type=int, help="Override RNG seed")
    p.add_argument("--log-level", default=None,
                   help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    overrides: dict = {"datasets": {"mitbih": {"enabled": True}}}
    if args.mitbih_path:
        overrides["datasets"]["mitbih"]["path"] = args.mitbih_path
    if args.records:
        overrides["datasets"]["mitbih"]["record_pattern"] = list(args.records)
    if args.max_events_per_record is not None:
        overrides["datasets"]["mitbih"]["max_events_per_record"] = args.max_events_per_record
    if args.out:
        overrides["output_dir"] = args.out
    if args.workers is not None:
        overrides.setdefault("runtime", {})["workers"] = args.workers
    if args.seed is not None:
        overrides.setdefault("runtime", {})["random_seed"] = args.seed
    if args.log_level:
        overrides.setdefault("runtime", {})["log_level"] = args.log_level

    cfg = load_config(args.config, overrides=overrides)
    if not cfg.datasets.get("mitbih", {}).get("path"):
        print(
            "error: --mitbih-path is required (or set datasets.mitbih.path in config)",
            file=sys.stderr,
        )
        return 2

    configure_logging(cfg.log_level)
    _log.info(
        "starting pipeline workers=%d output=%s",
        cfg.workers, os.path.abspath(cfg.output_dir),
    )
    t0 = time.time()
    written = Pipeline(cfg).run()
    dt = time.time() - t0
    summary = {
        "files_written": len(written),
        "elapsed_seconds": round(dt, 2),
        "output_dir": os.path.abspath(cfg.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
