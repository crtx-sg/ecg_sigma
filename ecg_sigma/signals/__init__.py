"""Signal-processing primitives: resampling, filtering, R-peak detection,
lead derivation and modality synthesis."""

from .processor import SignalProcessor
from .resampler import Resampler
from .lead_mapper import LeadMapper
from .peaks import detect_r_peaks
from .synthesis import ModalitiesSynthesizer
from .pacer import PacerGenerator, PacerConfig, pack_pacer_info, unpack_pacer_info

__all__ = [
    "SignalProcessor",
    "Resampler",
    "LeadMapper",
    "detect_r_peaks",
    "ModalitiesSynthesizer",
    "PacerGenerator",
    "PacerConfig",
    "pack_pacer_info",
    "unpack_pacer_info",
]
