"""Real-time imaging pipeline: ingest -> diagnostic gate -> (inference -> actuation)."""

from .quality import IntensityGate, QualityReport, QualityStage, to_gray
from .sources import CameraSource, Frame, FrameSource, SyntheticSource

__version__ = "0.1.0"
