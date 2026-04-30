"""Map heterogeneous lead sets to the canonical 7-lead output.

Output leads (see :mod:`ecg_sigma.schema`):
    ECG1 = I, ECG2 = II, ECG3 = III, aVR, aVL, aVF, vVX (precordial-like)

Einthoven / Goldberger relations:
    III  = II - I
    aVR  = -(I + II) / 2
    aVL  = (I - III) / 2  =  (3 I - 2 II) / 2     [equivalent forms]
    aVF  = (II + III) / 2 =  (2 II - I) / 2

The hard case is a 2-channel record with **only one** limb lead -- typical
for MIT-BIH (MLII + V1). Einthoven needs at least two limb leads, so for
those records we synthesise a plausible Lead I (low-pass + scaled inversion
of MLII) and tag every derived lead as synthetic. This is *not* clinically
correct, but it produces a 7-lead montage that downstream code can train
on without crashing on missing channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from ..schema import (
    DEFAULT_LEADS,
    METHOD_DIRECT,
    METHOD_EINTHOVEN,
    METHOD_RULE_BASED,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    ExtrasTag,
)


# Common WFDB lead-name aliases mapped onto the canonical labels we use.
_LIMB_ALIASES: Dict[str, str] = {
    "I":   "I",   "i":  "I",   "MLI":  "I",
    "II":  "II",  "ii": "II",  "MLII": "II",
    "III": "III", "iii": "III","MLIII": "III",
    "AVR": "aVR", "aVR": "aVR",
    "AVL": "aVL", "aVL": "aVL",
    "AVF": "aVF", "aVF": "aVF",
}

_PRECORDIAL_ALIASES = {
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
}


@dataclass
class LeadResult:
    """Per-lead output: signal + provenance tag."""

    signal: np.ndarray
    extras: ExtrasTag


@dataclass
class MappedLeads:
    """Container returned by :meth:`LeadMapper.map`."""

    leads: Dict[str, LeadResult]

    def array(self, order: Tuple[str, ...] = DEFAULT_LEADS) -> np.ndarray:
        """Return a (n_leads, n_samples) float32 array in canonical order."""
        return np.stack([self.leads[name].signal for name in order]).astype(np.float32)


class LeadMapper:
    """Lead derivation engine.

    Usage::

        mapper = LeadMapper()
        out = mapper.map(channels={"MLII": x_mlii, "V1": x_v1}, fs=200.0)
        ecg = out.array()                 # shape (7, 2400)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Entry-point
    # ------------------------------------------------------------------ #
    def map(
        self,
        channels: Dict[str, np.ndarray],
        fs: float,
    ) -> MappedLeads:
        """Map an arbitrary input lead set to the canonical 7-lead output.

        Parameters
        ----------
        channels:
            Dict ``{lead_name: 1-D array}`` *already resampled to* ``fs``.
            Lead names are matched case-insensitively against
            :data:`_LIMB_ALIASES` and :data:`_PRECORDIAL_ALIASES`.
        fs:
            Sampling frequency, in Hz, of the input channels.
        """
        limb, precordial = self._normalise(channels)
        if not limb and not precordial:
            raise ValueError("no recognisable ECG channels in input")

        # Reference length: every output lead must have this many samples.
        # Use the first input we see; all inputs should already be aligned.
        any_signal = next(iter({**limb, **precordial}.values()))
        n = any_signal.size

        I, II, III, src_I, src_II = self._resolve_limb(limb, n=n, fs=fs)

        leads: Dict[str, LeadResult] = {}
        leads["ECG1"] = LeadResult(I.astype(np.float32), src_I)
        leads["ECG2"] = LeadResult(II.astype(np.float32), src_II)

        # III: prefer real if provided, else derive.
        if "III" in limb:
            leads["ECG3"] = LeadResult(
                limb["III"].astype(np.float32),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead III"),
            )
        else:
            leads["ECG3"] = LeadResult(
                (II - I).astype(np.float32),
                ExtrasTag(
                    SOURCE_SYNTHETIC if (src_I.source == SOURCE_SYNTHETIC
                                         or src_II.source == SOURCE_SYNTHETIC)
                    else SOURCE_SYNTHETIC,  # III is always derived if not present
                    METHOD_EINTHOVEN,
                    "III = II - I",
                ),
            )

        # aVR/aVL/aVF: prefer real if provided, else Goldberger.
        for name, formula, key in (
            ("aVR", -(I + II) / 2.0, "aVR"),
            ("aVL", (I - leads["ECG3"].signal.astype(np.float64)) / 2.0, "aVL"),
            ("aVF", (II + leads["ECG3"].signal.astype(np.float64)) / 2.0, "aVF"),
        ):
            if key in limb:
                leads[name] = LeadResult(
                    limb[key].astype(np.float32),
                    ExtrasTag(SOURCE_REAL, METHOD_DIRECT, f"from input lead {key}"),
                )
            else:
                leads[name] = LeadResult(
                    formula.astype(np.float32),
                    ExtrasTag(
                        SOURCE_SYNTHETIC,
                        METHOD_EINTHOVEN,
                        f"{name} from Goldberger relations",
                    ),
                )

        # vVX: prefer V1 (the most common precordial in MIT-BIH); else any V*;
        # else synthesise from limb-lead.
        v_lead, v_name = self._select_precordial(precordial)
        if v_lead is not None:
            leads["vVX"] = LeadResult(
                v_lead.astype(np.float32),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, f"from input lead {v_name}"),
            )
        else:
            leads["vVX"] = LeadResult(
                self._synth_v(II, fs).astype(np.float32),
                ExtrasTag(
                    SOURCE_SYNTHETIC,
                    METHOD_RULE_BASED,
                    "synthesised V-equivalent from Lead II "
                    "(low-pass + scaled inversion)",
                ),
            )

        return MappedLeads(leads=leads)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(
        channels: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Split inputs into (limb-leads, precordial-leads), canonicalising names."""
        limb: Dict[str, np.ndarray] = {}
        precordial: Dict[str, np.ndarray] = {}
        for raw_name, sig in channels.items():
            sig1 = np.asarray(sig, dtype=np.float64).reshape(-1)
            key = raw_name.strip()
            if key in _LIMB_ALIASES:
                limb[_LIMB_ALIASES[key]] = sig1
            elif key in _PRECORDIAL_ALIASES:
                precordial[_PRECORDIAL_ALIASES[key]] = sig1
            else:
                # Unknown lead: try a fuzzy match on prefixes (handles e.g.
                # "ML II " or other oddities). Otherwise drop it.
                upper = key.upper().replace(" ", "")
                if upper in _LIMB_ALIASES:
                    limb[_LIMB_ALIASES[upper]] = sig1
                elif upper in _PRECORDIAL_ALIASES:
                    precordial[_PRECORDIAL_ALIASES[upper]] = sig1
        return limb, precordial

    def _resolve_limb(
        self, limb: Dict[str, np.ndarray], n: int, fs: float,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], ExtrasTag, ExtrasTag]:
        """Return (I, II, III?, extras_I, extras_II).

        Strategies, in priority order:
          1. I and II both present -> use directly.
          2. I and III present -> II = I + III.
          3. II and III present -> I = II - III.
          4. Only II present (e.g. MIT-BIH MLII) -> synthesise I.
          5. Only I present -> synthesise II.
          6. Only III present (rare) -> synthesise II, derive I.
        """
        if "I" in limb and "II" in limb:
            return (
                limb["I"],
                limb["II"],
                limb.get("III"),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead I"),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead II"),
            )
        if "I" in limb and "III" in limb:
            II = limb["I"] + limb["III"]
            return (
                limb["I"], II, limb["III"],
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead I"),
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_EINTHOVEN, "II = I + III"),
            )
        if "II" in limb and "III" in limb:
            I = limb["II"] - limb["III"]
            return (
                I, limb["II"], limb["III"],
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_EINTHOVEN, "I = II - III"),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead II"),
            )
        # Only one limb lead: synthesise the missing one with a clearly-tagged
        # rule-based transform (low-pass + scaled inversion preserves rate
        # and morphology while breaking exact equality).
        if "II" in limb:
            I = self._synth_partner_lead(limb["II"], fs)
            return (
                I, limb["II"], None,
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED,
                          "I synthesised from II (1-limb-lead record)"),
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead II"),
            )
        if "I" in limb:
            II = self._synth_partner_lead(limb["I"], fs)
            return (
                limb["I"], II, None,
                ExtrasTag(SOURCE_REAL, METHOD_DIRECT, "from input lead I"),
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED,
                          "II synthesised from I (1-limb-lead record)"),
            )
        if "III" in limb:
            II = self._synth_partner_lead(limb["III"], fs)
            I = II - limb["III"]
            return (
                I, II, limb["III"],
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED,
                          "I synthesised from III (1-limb-lead record)"),
                ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED,
                          "II synthesised from III (1-limb-lead record)"),
            )
        # No limb leads at all: build I/II from a precordial substitute.
        # Defer to caller; we do not arrive here when input has only V-leads
        # because the public ``map`` enforces at least one usable channel.
        zeros = np.zeros(n, dtype=np.float64)
        return (
            zeros, zeros, None,
            ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED, "no limb leads -- zero-filled"),
            ExtrasTag(SOURCE_SYNTHETIC, METHOD_RULE_BASED, "no limb leads -- zero-filled"),
        )

    # Lead synthesis primitives ----------------------------------------------
    @staticmethod
    def _synth_partner_lead(x: np.ndarray, fs: float) -> np.ndarray:
        """Build a plausible *partner* limb lead from a single input.

        We cannot recover a true second limb lead from one channel; instead
        we produce a phase-shifted, scaled, low-passed version. This keeps
        the heart-rate content intact and yields a non-degenerate III/aVR/aVL/aVF
        when fed through Einthoven. The output is *not* clinically accurate
        and is always tagged synthetic in the extras metadata.
        """
        nyq = 0.5 * fs
        cutoff = min(20.0, 0.45 * nyq)
        sos = sp_signal.butter(4, cutoff / nyq, btype="low", output="sos")
        smoothed = sp_signal.sosfiltfilt(sos, x)
        # Lead I is typically smaller-amplitude and roughly inverted relative
        # to MLII for inferior axes; 0.6 * inverted, plus a small lag
        # (3 ms) keeps things distinct.
        lag = max(1, int(round(0.003 * fs)))
        partner = -0.6 * np.roll(smoothed, lag)
        partner[:lag] = partner[lag]  # avoid wrap-around artefacts
        return partner

    @staticmethod
    def _synth_v(x: np.ndarray, fs: float) -> np.ndarray:
        """Synthesise a V-lead-like signal from limb Lead II.

        V-leads typically show sharper, more positive R-waves than MLII.
        We approximate this with a high-pass at 1 Hz + mild squaring of
        the QRS-band content. The result is *not* a real V-lead and is
        tagged synthetic.
        """
        nyq = 0.5 * fs
        sos_hp = sp_signal.butter(2, 1.0 / nyq, btype="high", output="sos")
        sos_qrs = sp_signal.butter(4, [5.0 / nyq, 20.0 / nyq], btype="band",
                                   output="sos")
        hp = sp_signal.sosfiltfilt(sos_hp, x)
        qrs = sp_signal.sosfiltfilt(sos_qrs, x)
        # Soft non-linearity emphasising QRS positivity.
        emphasis = np.sign(qrs) * (np.abs(qrs) ** 1.2) * 0.4
        return hp + emphasis

    @staticmethod
    def _select_precordial(
        precordial: Dict[str, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Pick the best available precordial lead for vVX. Prefer V1, then V2..V6."""
        for k in ("V1", "V2", "V3", "V4", "V5", "V6"):
            if k in precordial:
                return precordial[k], k
        return None, None
