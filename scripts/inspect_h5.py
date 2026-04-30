"""Pretty-print the structure and key attributes of a written HDF5 file.

Usage::

    python -m scripts.inspect_h5 path/to/file.h5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py  # noqa: E402
import numpy as np  # noqa: E402


def _attr_to_py(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _attr_to_py(value.item())
        if value.dtype.kind in ("S", "O"):
            return [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                for v in value
            ]
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _scalar(ds) -> object:
    return _attr_to_py(np.array(ds))


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _pretty_attrs(node) -> dict:
    return {k: _attr_to_py(v) for k, v in node.attrs.items()}


def _vital_to_dict(vg) -> dict:
    """Read the vitals sub-group's child datasets into a plain dict."""
    return {
        "value": _scalar(vg["value"]),
        "units": _scalar(vg["units"]),
        "timestamp": _scalar(vg["timestamp"]),
        "extras": _maybe_json(_scalar(vg["extras"])),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect an ecg_sigma HDF5 file")
    p.add_argument("path")
    p.add_argument("--max-events", type=int, default=2)
    args = p.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"file not found: {args.path}", file=sys.stderr)
        return 2

    with h5py.File(args.path, "r") as f:
        out = {"file": os.path.abspath(args.path), "metadata": {}, "events": []}
        if "metadata" in f:
            out["metadata"] = _pretty_attrs(f["metadata"])

        events = sorted(k for k in f.keys() if k.startswith("event_"))
        for name in events[: args.max_events]:
            grp = f[name]
            evt = {
                "name": name,
                "attrs": _pretty_attrs(grp),
                "ecg": {},
                "ppg": {},
                "resp": {},
                "vitals": {},
            }
            if "timestamp" in grp:
                evt["timestamp"] = float(np.array(grp["timestamp"]))
            if "uuid" in grp:
                evt["uuid"] = _attr_to_py(np.array(grp["uuid"]))

            ecg_grp = grp.get("ecg")
            if ecg_grp is not None:
                for lead in ecg_grp:
                    if lead == "extras":
                        evt["ecg"]["extras"] = _maybe_json(_scalar(ecg_grp["extras"]))
                    else:
                        evt["ecg"][lead] = {"shape": list(ecg_grp[lead].shape)}

            ppg_grp = grp.get("ppg")
            if ppg_grp is not None:
                for sig in ppg_grp:
                    if sig == "extras":
                        evt["ppg"]["extras"] = _maybe_json(_scalar(ppg_grp["extras"]))
                    else:
                        evt["ppg"][sig] = {"shape": list(ppg_grp[sig].shape)}

            resp_grp = grp.get("resp")
            if resp_grp is not None:
                for sig in resp_grp:
                    if sig == "extras":
                        evt["resp"]["extras"] = _maybe_json(_scalar(resp_grp["extras"]))
                    else:
                        evt["resp"][sig] = {"shape": list(resp_grp[sig].shape)}

            vitals_grp = grp.get("vitals")
            if vitals_grp is not None:
                for vname in vitals_grp:
                    evt["vitals"][vname] = _vital_to_dict(vitals_grp[vname])

            out["events"].append(evt)
        out["n_events_total"] = len(events)

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
