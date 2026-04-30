"""Validate one or more ecg_sigma HDF5 files against the strict schema.

Usage::

    # Validate a single file
    python -m scripts.validate_h5 out/mitbih/100_2025-01.h5

    # Validate every file under a directory tree
    python -m scripts.validate_h5 out/

    # Skip the per-vital history-integrity check
    python -m scripts.validate_h5 out/ --no-verify-history

Exits non-zero if any file fails. Soft warnings (e.g. low quality score)
are printed but do not fail the run.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecg_sigma.validation import (  # noqa: E402
    ValidationError,
    validate_pipeline_output,
)


def _expand(targets: list[str]) -> list[str]:
    out: list[str] = []
    for t in targets:
        if os.path.isdir(t):
            out.extend(sorted(glob.glob(os.path.join(t, "**", "*.h5"), recursive=True)))
        else:
            out.append(t)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate ecg_sigma HDF5 outputs")
    p.add_argument("paths", nargs="+", help="One or more files or directories")
    p.add_argument(
        "--verify-history", dest="verify_history", action="store_true",
        default=True,
        help="Verify per-vital history integrity (default: on)",
    )
    p.add_argument(
        "--no-verify-history", dest="verify_history", action="store_false",
        help="Skip the history-integrity check (faster on large runs)",
    )
    args = p.parse_args(argv)

    targets = _expand(args.paths)
    if not targets:
        print("no files matched", file=sys.stderr)
        return 2

    failures = 0
    for path in targets:
        try:
            warnings = validate_pipeline_output(
                path, verify_history=args.verify_history,
            )
        except ValidationError as exc:
            failures += 1
            print(f"FAIL {path}\n      {exc}")
            continue
        suffix = (
            f" ({len(warnings)} warning{'s' if len(warnings) != 1 else ''})"
            if warnings else ""
        )
        print(f"OK   {path}{suffix}")
        for w in warnings:
            print(f"     - {w}")

    print(f"\n{len(targets) - failures} of {len(targets)} files OK", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
