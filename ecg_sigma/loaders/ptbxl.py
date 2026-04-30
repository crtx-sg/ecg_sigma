"""PTB-XL adapter.

PTB-XL distributes signal records under e.g.::

    {root}/
        records100/00000/00001_lr.{dat,hea}
        records500/00000/00001_hr.{dat,hea}
        ptbxl_database.csv
        scp_statements.csv

We use the metadata CSV for the patient/scp-codes label set and the
companion record file for the actual ECG. By default we pick the
``records500`` (500 Hz) variant for fidelity.
"""

from __future__ import annotations

import ast
import os
from typing import Iterator, List, Optional

import numpy as np

from ..utils.logging import get_logger
from .base import DatasetLoader, PatientRecord, RhythmAnnotation

_log = get_logger(__name__)


class PTBXLLoader(DatasetLoader):
    """Adapter for the PTB-XL 12-lead ECG database (PhysioNet)."""

    name = "ptbxl"

    def __init__(
        self,
        root: str,
        sampling_rate: int = 500,
        max_records: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(root, **kwargs)
        if sampling_rate not in (100, 500):
            raise ValueError("PTB-XL sampling rate must be 100 or 500")
        self.sampling_rate = sampling_rate
        self.max_records = max_records

    # ------------------------------------------------------------------ #
    # Listing
    # ------------------------------------------------------------------ #
    def list_record_ids(self) -> List[str]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for PTB-XL loader") from exc

        csv_path = os.path.join(self.root, "ptbxl_database.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path, index_col="ecg_id")
        col = "filename_hr" if self.sampling_rate == 500 else "filename_lr"
        ids = [f"{int(idx)}|{rel}" for idx, rel in df[col].items()]
        if self.max_records:
            ids = ids[: self.max_records]
        return ids

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    def iter_records(self) -> Iterator[PatientRecord]:
        try:
            import pandas as pd
            import wfdb
        except ImportError as exc:
            raise RuntimeError(
                "PTB-XL loader requires pandas and wfdb"
            ) from exc

        csv_path = os.path.join(self.root, "ptbxl_database.csv")
        if not os.path.exists(csv_path):
            _log.warning("PTB-XL ptbxl_database.csv not found at %s", csv_path)
            return
        df = pd.read_csv(csv_path, index_col="ecg_id")
        col = "filename_hr" if self.sampling_rate == 500 else "filename_lr"

        count = 0
        for ecg_id, row in df.iterrows():
            if self.max_records is not None and count >= self.max_records:
                break
            rel_path = row[col]
            full = os.path.join(self.root, rel_path)
            if not (os.path.exists(full + ".hea") and os.path.exists(full + ".dat")):
                _log.debug("PTB-XL record %s not found on disk; skipping", rel_path)
                continue
            try:
                record = wfdb.rdrecord(full)
            except Exception as exc:  # noqa: BLE001
                _log.warning("PTB-XL: could not read %s: %s", rel_path, exc)
                continue

            sig = np.asarray(record.p_signal, dtype=np.float64)
            sig_names = list(record.sig_name) if record.sig_name else [
                f"CH{i}" for i in range(sig.shape[1])
            ]
            signals = {name: sig[:, i] for i, name in enumerate(sig_names)}

            scp_codes = self._parse_scp(row.get("scp_codes", "{}"))
            patient_id = f"PTBXL-{int(row['patient_id'])}"
            yield PatientRecord(
                patient_id=patient_id,
                dataset=self.name,
                fs=float(record.fs),
                signals=signals,
                beat_annotations=[],
                rhythm_annotations=[
                    RhythmAnnotation(
                        labels=tuple(scp_codes.keys()),
                        onset_sample=0,
                        offset_sample=int(sig.shape[0]),
                        confidence=float(max(scp_codes.values()) / 100.0)
                        if scp_codes else 1.0,
                    )
                ],
                base_time_epoch=0,
                metadata={
                    "ecg_id": int(ecg_id),
                    "age": int(row.get("age", -1)) if not _is_nan(row.get("age")) else None,
                    "sex": int(row.get("sex", -1)) if not _is_nan(row.get("sex")) else None,
                    "scp_codes": scp_codes,
                    "source_record_path": full,
                },
            )
            count += 1

    @staticmethod
    def _parse_scp(value) -> dict:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _is_nan(x) -> bool:
    try:
        return x != x   # NaN-safe in pure Python
    except TypeError:
        return False
