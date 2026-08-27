"""Real-time imaging pipeline: ingest -> diagnostic gate -> segmentation -> DMD."""

from .background import BackgroundStage, NumpyBackgroundSubtractor, make_subtractor
from .closedloop import ClosedLoopStage, CycleResult, NumpyController, make_controller
from .native import HAVE_NATIVE, backend_name, describe
from .pipeline import Metrics, Pipeline
from .quality import IntensityGate, QualityReport, QualityStage, to_gray
from .segmentation import NumpySegmenter, SegmentationResult
from .sources import CameraSource, Frame, FrameSource, SyntheticSource

__version__ = "0.3.0"

__all__ = [
    "BackgroundStage",
    "CameraSource",
    "ClosedLoopStage",
    "CycleResult",
    "Frame",
    "FrameSource",
    "HAVE_NATIVE",
    "IntensityGate",
    "Metrics",
    "NumpyBackgroundSubtractor",
    "NumpyController",
    "NumpySegmenter",
    "Pipeline",
    "QualityReport",
    "QualityStage",
    "SegmentationResult",
    "SyntheticSource",
    "backend_name",
    "describe",
    "make_controller",
    "make_subtractor",
    "to_gray",
    "__version__",
]
