"""Command-line runner: ``clpipe --frames 300`` or ``python -m clpipe``."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .pipeline import Pipeline
from .quality import IntensityGate, QualityStage
from .sources import CameraSource, FrameSource, SyntheticSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clpipe",
        description="Run the ingest -> diagnostic-gate pipeline and report timing.",
    )
    parser.add_argument("--source", choices=("synthetic", "camera"), default="synthetic")
    parser.add_argument("--device", default="0", help="camera index or video path")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=300,
                        help="number of frames to run; 0 means until interrupted")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--no-drop", action="store_true",
                        help="block the producer instead of dropping stale frames")
    parser.add_argument("--low", type=float, default=10.0, help="reject below this mean intensity")
    parser.add_argument("--high", type=float, default=245.0, help="reject above this mean intensity")
    parser.add_argument("--free-run", action="store_true",
                        help="synthetic source only: generate as fast as possible")
    parser.add_argument("--verbose", action="store_true", help="log every rejected frame")
    return parser


def build_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "camera":
        device: int | str = int(args.device) if args.device.isdigit() else args.device
        return CameraSource(device=device, width=args.width, height=args.height, fps=args.fps)

    return SyntheticSource(
        width=args.width,
        height=args.height,
        fps=args.fps,
        n_frames=args.frames or None,
        realtime=not args.free_run,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    stage = QualityStage(gate=IntensityGate(low=args.low, high=args.high))
    if args.verbose:
        original = stage.gate

        def _wrapped(frame):
            report = original(frame)
            if not report.ok:
                print(f"  frame {frame.index:5d} rejected: {report.reason} "
                      f"(mean={report.mean_intensity:.1f})", file=sys.stderr)
            return report

        stage.gate = _wrapped  # type: ignore[assignment]

    source = build_source(args)
    pipeline = Pipeline(
        source,
        handler=stage,
        queue_size=args.queue_size,
        drop_when_full=not args.no_drop,
    )

    print(f"running {args.source} source, queue depth {args.queue_size}, "
          f"drop_when_full={not args.no_drop}")
    metrics = pipeline.run()

    print(metrics.summary())
    print(f"gate: accepted={stage.accepted} rejected={stage.rejected} {stage.reasons or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
