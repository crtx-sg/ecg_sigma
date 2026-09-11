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
    "(N": NORMAL_SINUS,   # Normal sinus rhythm
    "(NSR": NORMAL_SINUS,
    "(SBR": BRADYCARDIA,  # Sinus bradycardia -- the ONLY bradycardia code
    "(B": PVC,            # Ventricular BIGEMINY (not bradycardia): a PVC pattern
    "(T": PVC,            # Ventricular TRIGEMINY (not tachycardia): a PVC pattern
    "(SVTA": TACHYCARDIA, # Supraventricular tachyarrhythmia
    "(VT": VTACH,         # Ventricular tachycardia
    "(VFL": VFIB,         # Ventricular flutter
    "(VF": VFIB,
    "(AFIB": AFIB,        # Atrial fibrillation
    "(AFL": AFIB,         # Atrial flutter (close family)
    "(AB": PAC,           # Atrial bigeminy -- a PAC pattern
    "(NOD": OTHER,        # Nodal (A-V junctional) rhythm -- no unified label
    "(IVR": OTHER,        # Idioventricular rhythm
    "(P": PACED,          # Paced rhythm
    "(PREX": OTHER,       # Pre-excitation (WPW)
    "(BII": OTHER,        # 2-degree heart block
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

# Priority order used whenever two labels describe the same window: pick the
# most clinically urgent. Used for PTB-XL's multi-label records and for
# reconciling a MIT-BIH beat label against its background rhythm.
# Ordered by what an ICU alarm would announce, most urgent first: lethal
# ventricular rhythms, then the rate/pause alarms, then atrial rhythms, then
# beat morphology, then baseline. Rate alarms deliberately outrank morphology
# -- a PAC inside a sinus-bradycardia strip is still a bradycardia alarm.
# Events keep their raw beat symbol and rhythm context as separate attributes,
# so a consumer that wants morphology-first labels can re-derive them.
CONDITION_PRIORITY = (
    VFIB, VTACH, PAUSE, BRADYCARDIA, TACHYCARDIA, AFIB, MI,
    PVC, PAC, LBBB, RBBB, PACED, NORMAL_SINUS, OTHER,
)
PTBXL_PRIORITY = CONDITION_PRIORITY   # back-compat alias


def map_mitbih_beat(symbol: str) -> str:
    """Map a single WFDB beat symbol to a unified condition label."""
    return MITBIH_BEAT_MAP.get(symbol, OTHER)


def map_mitbih_rhythm(aux_note: str) -> Optional[str]:
    """Map an MIT-BIH rhythm aux note (e.g. '(AFIB') to a unified label.

    Returns ``None`` for an empty string or an unrecognised note so that
    callers can decide whether to fall back to a beat-based label.

    WFDB pads aux notes to an even byte count with a trailing NUL, so a
    plain ``str.strip()`` leaves ``'(AFIB\x00'`` intact and every lookup
    misses. Strip NULs (and other control padding) explicitly.
    """
    if not aux_note:
        return None
    cleaned = aux_note.strip().strip("\x00").strip()
    if not cleaned:
        return None
    key = cleaned.split()[0]
    return MITBIH_RHYTHM_MAP.get(key)


def resolve_condition(*candidates: Optional[str]) -> str:
    """Reduce several condition labels for one window to the most urgent.

    A MIT-BIH beat carries two independent descriptions: its own morphology
    (``V`` -> PVC) and the background rhythm it sits in (``(AFIB`` -> AFIB).
    Letting the rhythm blindly win erases every ectopic beat inside a sinus
    strip; letting the beat win erases VT/AFIB runs. Ranking both through
    :data:`CONDITION_PRIORITY` keeps whichever an alarm would actually
    announce -- a PVC during sinus stays PVC, a ``V`` beat inside a ``(VT``
    run becomes VTACH.
    """
    present = {c for c in candidates if c}
    if not present:
        return OTHER
    for label in CONDITION_PRIORITY:
        if label in present:
            return label
    return OTHER


def map_ptbxl_labels(scp_codes: Iterable[str]) -> str:
    """Map an iterable of PTB-XL SCP codes to a unified label.

    PTB-XL is multi-label per record. We translate each code into the
    unified vocabulary and then resolve the collision via :data:`CONDITION_PRIORITY`.
    """
    candidates = {PTBXL_SCP_MAP.get(c, OTHER) for c in scp_codes}
    if not candidates:
        return OTHER
    for label in CONDITION_PRIORITY:
        if label in candidates:
            return label
    return OTHER
