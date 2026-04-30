"""MIT-BIH Arrhythmia Database adapter.

Layout expected::

    {root}/
        100.dat / 100.hea / 100.atr
        101.dat / 101.hea / 101.atr
        ...

We rely on `wfdb` for the heavy lifting. WFDB returns:
    - ``sig``: ``(n_samples, n_channels)`` float array (already in mV)
    - ``fields``: dict including ``sig_name`` (lead labels) and ``fs``
    - ``annotation``: parsed ``.atr`` annotation file
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

import numpy as np

from ..utils.logging import get_logger
from .base import BeatAnnotation, DatasetLoader, PatientRecord

_log = get_logger(__name__)


class MITBIHLoader(DatasetLoader):
    """Adapter for the MIT-BIH Arrhythmia Database (PhysioNet)."""

    name = "mitbih"

    def __init__(
        self,
        root: str,
        record_pattern: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(root, **kwargs)
        self.record_pattern = record_pattern

    # ------------------------------------------------------------------ #
    # Listing
    # ------------------------------------------------------------------ #
    def list_record_ids(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.root)
            if f.endswith(".hea")
        )
        if self.record_pattern:
            allowed = set(self.record_pattern)
            names = [n for n in names if n in allowed]
        return names

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    def iter_records(self) -> Iterator[PatientRecord]:
        try:
            import wfdb  # imported lazily so the package is optional at install time
        except ImportError as exc:
            raise RuntimeError(
                "wfdb is required to load MIT-BIH; pip install wfdb"
            ) from exc

        for record_id in self.list_record_ids():
            try:
                record = wfdb.rdrecord(os.path.join(self.root, record_id))
                ann = wfdb.rdann(os.path.join(self.root, record_id), "atr")
            except Exception as exc:  # noqa: BLE001 - wfdb raises a variety of errors
                _log.warning("skipping record %s: %s", record_id, exc)
                continue

            sig = np.asarray(record.p_signal, dtype=np.float64)
            if sig.ndim != 2:
                _log.warning("record %s has unexpected shape %s", record_id, sig.shape)
                continue
            sig_names = list(record.sig_name) if record.sig_name else [
                f"CH{i}" for i in range(sig.shape[1])
            ]
            signals = {name: sig[:, i] for i, name in enumerate(sig_names)}

            beat_anns = self._parse_annotations(ann)
            yield PatientRecord(
                patient_id=record_id,
                dataset=self.name,
                fs=float(record.fs),
                signals=signals,
                beat_annotations=beat_anns,
                rhythm_annotations=[],
                base_time_epoch=0,
                metadata={
                    "source_record_path": os.path.join(self.root, record_id),
                    "n_samples_raw": int(sig.shape[0]),
                    "channels_raw": sig_names,
                    "comments": list(getattr(record, "comments", []) or []),
                },
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_annotations(ann) -> list[BeatAnnotation]:
        out: list[BeatAnnotation] = []
        n = len(ann.sample) if ann.sample is not None else 0
        for i in range(n):
            sample = int(ann.sample[i])
            symbol = ann.symbol[i] if ann.symbol else ""
            aux = ""
            if ann.aux_note is not None and i < len(ann.aux_note):
                aux = (ann.aux_note[i] or "").strip()
            out.append(BeatAnnotation(sample=sample, symbol=symbol, aux_note=aux))
        return out
