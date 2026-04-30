"""Event extraction abstraction.

An ``Event`` is the unit the downstream pipeline operates on: a single
12-second window centred on something we want to alarm/train on. Every
loader can produce events through one (or both) of these strategies:

  * :class:`BeatBasedExtractor` -- for datasets with sample-level
    annotations (MIT-BIH, INCART).
  * :class:`RhythmBasedExtractor` -- for datasets with record-level
    rhythm/diagnostic labels (PTB-XL).

The extractor returns a list of :class:`Event` objects; it does *not*
crop signals -- the pipeline does that, after lead derivation/resampling,
so the index math lives in one place.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List


@dataclass
class Event:
    """A single window-of-interest within a record.

    Attributes
    ----------
    onset_sample:
        Sample index of the event centre, in *source-fs* coordinates.
    condition:
        Unified condition label (see :mod:`ecg_sigma.conditions`).
    source_label:
        Original label from the dataset (for traceability).
    metadata:
        Free-form extras (rhythm context, beat morphology, ...).
    """

    onset_sample: int
    condition: str
    source_label: str = ""
    metadata: dict = field(default_factory=dict)


class EventExtractor(abc.ABC):
    """Strategy that turns a :class:`PatientRecord` into a list of events."""

    @abc.abstractmethod
    def extract(self, record) -> List[Event]:  # PatientRecord avoided to dodge circular import
        raise NotImplementedError
