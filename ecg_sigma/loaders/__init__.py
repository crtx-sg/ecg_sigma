"""Dataset loaders. Each adapter yields :class:`PatientRecord` objects."""

from .base import DatasetLoader, PatientRecord
from .mitbih import MITBIHLoader
from .ptbxl import PTBXLLoader
from .incart import INCARTLoader

__all__ = [
    "DatasetLoader",
    "PatientRecord",
    "MITBIHLoader",
    "PTBXLLoader",
    "INCARTLoader",
]
