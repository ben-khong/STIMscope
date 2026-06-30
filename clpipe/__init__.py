"""Real-time imaging pipeline: ingest -> diagnostic gate -> (inference -> actuation)."""

from .pipeline import Metrics, Pipeline
from .quality import IntensityGate, QualityReport, QualityStage, to_gray
from .sources import CameraSource, Frame, FrameSource, SyntheticSource

__version__ = "0.1.0"

__all__ = [
    "CameraSource",
    "Frame",
    "FrameSource",
    "IntensityGate",
    "Metrics",
    "Pipeline",
    "QualityReport",
    "QualityStage",
    "SyntheticSource",
    "to_gray",
    "__version__",
]
