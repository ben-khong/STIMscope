"""Real-time imaging pipeline: ingest -> diagnostic gate -> (inference -> actuation)."""

from .background import BackgroundStage, NumpyBackgroundSubtractor, make_subtractor
from .native import HAVE_NATIVE, backend_name, describe
from .pipeline import Metrics, Pipeline
from .quality import IntensityGate, QualityReport, QualityStage, to_gray
from .sources import CameraSource, Frame, FrameSource, SyntheticSource

__version__ = "0.2.0"

__all__ = [
    "BackgroundStage",
    "CameraSource",
    "Frame",
    "FrameSource",
    "HAVE_NATIVE",
    "IntensityGate",
    "Metrics",
    "NumpyBackgroundSubtractor",
    "Pipeline",
    "QualityReport",
    "QualityStage",
    "SyntheticSource",
    "backend_name",
    "describe",
    "make_subtractor",
    "to_gray",
    "__version__",
]
