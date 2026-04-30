"""Dataset-loader abstraction.

Each adapter yields :class:`PatientRecord` objects. A "patient record" is
one continuous recording for one patient: the loader is the only layer
that knows about WFDB / CSV / file layouts.

The downstream pipeline only consumes :class:`PatientRecord` instances,
so adding a new dataset means writing one new adapter + a label-map
update in :mod:`ecg_sigma.conditions`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class BeatAnnotation:
    """One beat-level annotation."""

    sample: int          # sample index in the source-fs timeline
    symbol: str          # WFDB symbol (e.g. 'N', 'V', 'A')
    aux_note: str = ""   # WFDB aux_note (rhythm changes etc.)


@dataclass
class RhythmAnnotation:
    """A multi-label rhythm/diagnostic statement covering the whole record."""

    labels: Tuple[str, ...]
    onset_sample: int = 0
    offset_sample: Optional[int] = None  # None == end of record
    confidence: float = 1.0


@dataclass
class PatientRecord:
    """All the information one record provides to the pipeline.

    Attributes
    ----------
    patient_id:
        Stable identifier used in the output filename. *Must not* leak
        protected health information; for public datasets the record name
        (e.g. "100") is fine.
    dataset:
        Tag identifying the source dataset (e.g. ``"mitbih"``).
    fs:
        Sampling rate of the raw signals, in Hz.
    signals:
        Dict ``{lead_name: 1-D float array}`` at ``fs``.
    beat_annotations:
        Optional sequence of beat-level annotations (MIT-BIH style).
    rhythm_annotations:
        Optional sequence of rhythm-level annotations (PTB-XL style).
    base_time_epoch:
        Epoch seconds at sample 0. Datasets without absolute time use a
        deterministic anchor; see :func:`Pipeline._anchor_for`.
    metadata:
        Free-form per-record metadata (age, sex, ...). Written to the
        ``/metadata`` group attributes verbatim where compatible.
    """

    patient_id: str
    dataset: str
    fs: float
    signals: Dict[str, "object"]                       # numpy arrays
    beat_annotations: List[BeatAnnotation] = field(default_factory=list)
    rhythm_annotations: List[RhythmAnnotation] = field(default_factory=list)
    base_time_epoch: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


class DatasetLoader(abc.ABC):
    """ABC for dataset adapters.

    Concrete loaders must:

    1. Be cheap to instantiate (validation only); deferred work goes in
       :meth:`iter_records`.
    2. Be iterable repeatedly without side-effects (no global state).
    """

    name: str = "base"

    def __init__(self, root: str, **kwargs):
        self.root = root
        self.kwargs = kwargs

    @abc.abstractmethod
    def iter_records(self) -> Iterator[PatientRecord]:
        """Yield :class:`PatientRecord` objects."""
        raise NotImplementedError

    def list_record_ids(self) -> List[str]:
        """Best-effort listing for logging/filtering. Override if cheap."""
        return []
