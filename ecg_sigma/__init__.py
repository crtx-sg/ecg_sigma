"""ecg_sigma: convert public ECG datasets into ICU-style HDF5 with synthesised modalities."""

from .pipeline import Pipeline, PipelineConfig, load_config

__version__ = "0.1.0"
__all__ = ["Pipeline", "PipelineConfig", "load_config", "__version__"]
