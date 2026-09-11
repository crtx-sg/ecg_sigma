"""Beat-annotation -> event windows.

We turn each WFDB beat annotation into one event. To keep file sizes
manageable and avoid massive class imbalance, callers can:

  * restrict the included WFDB symbols (``include_symbols``)
  * cap the number of events per record (``max_events_per_record``)
  * stride to subsample dense streams (``stride``)

A beat carries two independent descriptions: its own morphology (``V``)
and the background rhythm it sits in (``(AFIB``). We reconcile them by
clinical urgency (:func:`ecg_sigma.conditions.resolve_condition`) rather
than letting either blindly win -- a PVC inside a sinus strip stays PVC,
while a ``V`` beat inside a ``(VT`` run is promoted to VTACH. This
matches what an alarm would actually announce. Both raw labels are kept
in ``Event.metadata`` for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..conditions import (
    OTHER,
    map_mitbih_beat,
    map_mitbih_rhythm,
    resolve_condition,
)
from .base import Event, EventExtractor


@dataclass
class BeatExtractorConfig:
    include_symbols: Optional[Tuple[str, ...]] = None
    max_events_per_record: Optional[int] = None
    stride: int = 1
    drop_other: bool = False         # if True, skip events that map to OTHER
    rng_seed: Optional[int] = None
    safety_margin_samples: int = 0   # drop beats < margin from either edge


class BeatBasedExtractor(EventExtractor):
    """Default extractor for MIT-BIH and INCART."""

    def __init__(self, cfg: BeatExtractorConfig):
        self.cfg = cfg

    def extract(self, record) -> List[Event]:
        if not record.beat_annotations:
            return []

        # Build a sorted list of (onset_sample, rhythm_label) for fast lookup.
        rhythm_segments = self._build_rhythm_segments(record.beat_annotations)

        keep_symbols = (
            set(self.cfg.include_symbols)
            if self.cfg.include_symbols is not None else None
        )

        # Pre-compute the valid sample range so we never produce events that
        # the pipeline would silently drop because the window goes off-record.
        first_signal = next(iter(record.signals.values()), None)
        n_total = int(getattr(first_signal, "size", 0))
        margin = max(0, int(self.cfg.safety_margin_samples))
        valid_lo = margin
        valid_hi = max(margin, n_total - margin)

        events: List[Event] = []
        for i, ann in enumerate(record.beat_annotations):
            if self.cfg.stride > 1 and (i % self.cfg.stride):
                continue
            if keep_symbols is not None and ann.symbol not in keep_symbols:
                continue
            if margin and (ann.sample < valid_lo or ann.sample > valid_hi):
                continue

            beat_cond = map_mitbih_beat(ann.symbol)
            rhythm_cond = self._rhythm_at(rhythm_segments, ann.sample)
            condition = resolve_condition(beat_cond, rhythm_cond)

            if self.cfg.drop_other and condition == OTHER:
                continue

            events.append(Event(
                onset_sample=int(ann.sample),
                condition=condition,
                source_label=ann.symbol,
                metadata={
                    "beat_symbol": ann.symbol,
                    "aux_note": ann.aux_note,
                    "beat_condition": beat_cond,
                    "rhythm_context": rhythm_cond or "",
                },
            ))

        if self.cfg.max_events_per_record is not None and \
                len(events) > self.cfg.max_events_per_record:
            events = self._subsample(events, self.cfg.max_events_per_record)
        return events

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_rhythm_segments(annotations: Iterable) -> List[Tuple[int, str]]:
        """Return [(onset_sample, rhythm_condition_or_None), ...] sorted ascending.

        WFDB encodes rhythm changes by attaching an ``aux_note`` like
        ``(AFIB`` to a beat; the rhythm holds until the next ``aux_note``.
        """
        segments: List[Tuple[int, str]] = []
        for ann in annotations:
            cond = map_mitbih_rhythm(ann.aux_note)
            if cond is None:
                continue
            segments.append((int(ann.sample), cond))
        return segments

    @staticmethod
    def _rhythm_at(segments: List[Tuple[int, str]], sample: int) -> Optional[str]:
        """Return the rhythm active at ``sample``, or None if no rhythm context."""
        if not segments:
            return None
        # Linear scan is fine: rhythm changes per record are O(10).
        active = None
        for onset, cond in segments:
            if onset > sample:
                break
            active = cond
        return active

    def _subsample(self, events: List[Event], cap: int) -> List[Event]:
        """Stratify by condition then take a deterministic subsample."""
        rng = np.random.default_rng(self.cfg.rng_seed or 0)
        # Group by condition and take roughly equal counts across classes.
        by_cond: dict[str, List[Event]] = {}
        for e in events:
            by_cond.setdefault(e.condition, []).append(e)
        per_class = max(1, cap // max(1, len(by_cond)))
        kept: List[Event] = []
        for cond, items in by_cond.items():
            if len(items) <= per_class:
                kept.extend(items)
                continue
            idx = rng.choice(len(items), size=per_class, replace=False)
            kept.extend(items[i] for i in sorted(idx))
        if len(kept) > cap:
            kept = kept[:cap]
        # Sort by onset for stable ordering.
        kept.sort(key=lambda e: e.onset_sample)
        return kept
