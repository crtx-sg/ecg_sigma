"""Rhythm-annotation -> event windows.

For datasets like PTB-XL the only label is record-level. By default we
emit one event per record at the centre. Callers can also request
``n_events`` evenly-spaced windows for longer recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..conditions import map_ptbxl_labels
from .base import Event, EventExtractor


@dataclass
class RhythmExtractorConfig:
    """One event per record by default; bump ``n_events`` for longer files."""

    n_events: int = 1
    label_set: str = "ptbxl"   # currently only PTB-XL is supported


class RhythmBasedExtractor(EventExtractor):

    def __init__(self, cfg: RhythmExtractorConfig):
        self.cfg = cfg

    def extract(self, record) -> List[Event]:
        if not record.rhythm_annotations:
            return []

        ann = record.rhythm_annotations[0]
        if self.cfg.label_set == "ptbxl":
            condition = map_ptbxl_labels(ann.labels)
        else:
            raise ValueError(f"unsupported label_set {self.cfg.label_set}")

        n_total = max(1, self.cfg.n_events)
        rec_len = ann.offset_sample or self._record_length(record)
        if rec_len <= 0:
            return []

        # Place events evenly inside the record, avoiding the edges.
        step = rec_len // (n_total + 1)
        positions = [(i + 1) * step for i in range(n_total)]

        return [
            Event(
                onset_sample=int(p),
                condition=condition,
                source_label="|".join(ann.labels),
                metadata={"rhythm_labels": list(ann.labels)},
            )
            for p in positions
        ]

    @staticmethod
    def _record_length(record) -> int:
        if not record.signals:
            return 0
        first = next(iter(record.signals.values()))
        return int(getattr(first, "size", 0))
