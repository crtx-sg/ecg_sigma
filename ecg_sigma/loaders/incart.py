"""St-Petersburg INCART Arrhythmia Database adapter.

Same WFDB layout as MIT-BIH, but typically 12-lead and 257 Hz. We reuse
the MIT-BIH adapter's parsing and only override the dataset name.
"""

from __future__ import annotations

from .mitbih import MITBIHLoader


class INCARTLoader(MITBIHLoader):
    """Adapter for the INCART 12-lead arrhythmia database (PhysioNet)."""

    name = "incart"
