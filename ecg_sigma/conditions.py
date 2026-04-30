"""Unified condition vocabulary and dataset-specific label mappings.

The arrhythmia/event "condition" is the model-relevant target. Each public
dataset uses its own coding system; this module normalises everything into a
single ALL-CAPS vocabulary so downstream code does not have to special-case
each dataset.

The vocabulary is intentionally compact. New dataset adapters should map
their labels into one of these values, falling back to ``OTHER`` when there
is no good fit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


# --------------------------------------------------------------------------- #
# Unified condition labels
# --------------------------------------------------------------------------- #
NORMAL_SINUS = "NORMAL_SINUS"
PVC = "PVC"
PAC = "PAC"
AFIB = "AFIB"
VTACH = "VTACH"
VFIB = "VFIB"
BRADYCARDIA = "BRADYCARDIA"
TACHYCARDIA = "TACHYCARDIA"
PAUSE = "PAUSE"
MI = "MI"
LBBB = "LBBB"
RBBB = "RBBB"
PACED = "PACED"
OTHER = "OTHER"

ALL_CONDITIONS = frozenset(
    {
        NORMAL_SINUS, PVC, PAC, AFIB, VTACH, VFIB, BRADYCARDIA,
        TACHYCARDIA, PAUSE, MI, LBBB, RBBB, PACED, OTHER,
    }
)


# --------------------------------------------------------------------------- #
# MIT-BIH / INCART beat annotations  -- WFDB symbol -> condition
# Reference: https://archive.physionet.org/physiobank/annotations.shtml
# --------------------------------------------------------------------------- #
MITBIH_BEAT_MAP: Dict[str, str] = {
    "N": NORMAL_SINUS,    # Normal beat
    ".": NORMAL_SINUS,    # Same as N (legacy)
    "L": LBBB,            # Left bundle-branch block beat
    "R": RBBB,            # Right bundle-branch block beat
    "A": PAC,             # Atrial premature contraction
    "a": PAC,             # Aberrated APC
    "J": PAC,             # Nodal (junctional) premature beat
    "S": PAC,             # Supraventricular premature/ectopic beat
    "V": PVC,             # Premature ventricular contraction
    "E": PVC,             # Ventricular escape
    "F": PVC,             # Fusion of ventricular and normal
    "!": VFIB,            # Ventricular flutter wave
    "[": VFIB,            # Start ventricular flutter/fibrillation
    "]": VFIB,            # End ventricular flutter/fibrillation
    "/": PACED,           # Paced beat
    "f": PACED,           # Fusion of paced and normal
    "j": PAC,             # Nodal (junctional) escape
    "Q": OTHER,           # Unclassifiable
    "?": OTHER,
}


# --------------------------------------------------------------------------- #
# MIT-BIH rhythm annotations  -- aux string -> condition
# Strings begin with '(' in the WFDB aux_note field.
# --------------------------------------------------------------------------- #
MITBIH_RHYTHM_MAP: Dict[str, str] = {
    "(N": NORMAL_SINUS,
    "(NSR": NORMAL_SINUS,
    "(SBR": BRADYCARDIA,
    "(B": BRADYCARDIA,
    "(SVTA": TACHYCARDIA,
    "(T": TACHYCARDIA,
    "(VT": VTACH,
    "(VFL": VFIB,
    "(VF": VFIB,
    "(AFIB": AFIB,
    "(AB": PAUSE,         # Atrial bigeminy treated as 'other rhythm'
    "(AFL": AFIB,         # Atrial flutter (close family)
    "(IVR": OTHER,
    "(P": PACED,
    "(PREX": OTHER,
    "(BII": OTHER,
}


# --------------------------------------------------------------------------- #
# PTB-XL diagnostic super-classes / SCP codes -> condition.
# PTB-XL provides multi-label diagnostic statements; we collapse them.
# --------------------------------------------------------------------------- #
PTBXL_SCP_MAP: Dict[str, str] = {
    "NORM": NORMAL_SINUS,
    "MI":   MI,
    "IMI":  MI,
    "AMI":  MI,
    "ASMI": MI,
    "ILMI": MI,
    "ALMI": MI,
    "INJAS": MI,
    "INJAL": MI,
    "INJIN": MI,
    "INJLA": MI,
    "STTC": OTHER,
    "ISC_": OTHER,
    "ISCAL": OTHER,
    "ISCAS": OTHER,
    "ISCIN": OTHER,
    "ISCLA": OTHER,
    "AFIB": AFIB,
    "AFLT": AFIB,
    "SR":   NORMAL_SINUS,
    "STACH": TACHYCARDIA,
    "SBRAD": BRADYCARDIA,
    "PSVT": TACHYCARDIA,
    "PVC":  PVC,
    "BIGU": PVC,
    "TRIGU": PVC,
    "PAC":  PAC,
    "SVARR": PAC,
    "LBBB": LBBB,
    "CLBBB": LBBB,
    "ILBBB": LBBB,
    "RBBB": RBBB,
    "CRBBB": RBBB,
    "IRBBB": RBBB,
    "PACE": PACED,
}

# Priority order when a record has multiple labels: pick the most clinically
# urgent label first.
PTBXL_PRIORITY = (
    VFIB, VTACH, MI, AFIB, PVC, PAC, LBBB, RBBB,
    TACHYCARDIA, BRADYCARDIA, PAUSE, PACED, NORMAL_SINUS, OTHER,
)


def map_mitbih_beat(symbol: str) -> str:
    """Map a single WFDB beat symbol to a unified condition label."""
    return MITBIH_BEAT_MAP.get(symbol, OTHER)


def map_mitbih_rhythm(aux_note: str) -> Optional[str]:
    """Map an MIT-BIH rhythm aux note (e.g. '(AFIB') to a unified label.

    Returns ``None`` for an empty string or an unrecognised note so that
    callers can decide whether to fall back to a beat-based label.
    """
    if not aux_note:
        return None
    key = aux_note.strip().split()[0]
    return MITBIH_RHYTHM_MAP.get(key)


def map_ptbxl_labels(scp_codes: Iterable[str]) -> str:
    """Map an iterable of PTB-XL SCP codes to a unified label.

    PTB-XL is multi-label per record. We translate each code into the
    unified vocabulary and then resolve the collision via :data:`PTBXL_PRIORITY`.
    """
    candidates = {PTBXL_SCP_MAP.get(c, OTHER) for c in scp_codes}
    if not candidates:
        return OTHER
    for label in PTBXL_PRIORITY:
        if label in candidates:
            return label
    return OTHER
