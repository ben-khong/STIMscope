"""Command-line runner: ``clpipe --frames 300`` or ``python -m clpipe``."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .background import BackgroundStage
from .closedloop import ClosedLoopStage
from .native import describe
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

    background = parser.add_argument_group("background subtraction")
    background.add_argument("--background", action="store_true",
                            help="run background subtraction on frames that pass the gate")
    background.add_argument("--alpha", type=float, default=0.05,
                            help="background model learning rate")
    background.add_argument("--bg-threshold", type=int, default=25,
                            help="per-pixel deviation counted as foreground")
    background.add_argument("--backend", choices=("numpy", "cpp", "cuda"), default=None,
                            help="force a backend instead of picking the best available")

    loop = parser.add_argument_group("closed loop")
    loop.add_argument("--closed-loop", action="store_true",
                      help="segment accepted frames and drive the DMD (implies --background)")
    loop.add_argument("--segmenter", default="builtin",
                      help="'builtin' or a path to a .onnx segmentation model")
    loop.add_argument("--activation-threshold", type=int, default=128,
                      help="confidence at or above which a pixel is a stimulation target")
    loop.add_argument("--no-motion-veto", action="store_true",
                      help="stimulate on segmentation alone, ignoring the motion mask")
    loop.add_argument("--open-loop", action="store_true",
                      help="run every stage but do not feed the pattern back to the sample")
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

    # Built first: the closed loop needs a handle on the sample to feed the
    # projected pattern back into it.
    source = build_source(args)

    loop: ClosedLoopStage | None = None
    if args.closed_loop:
        sample = source if (args.source == "synthetic" and not args.open_loop) else None
        loop = ClosedLoopStage(
            segmenter=args.segmenter,
            activation_threshold=args.activation_threshold,
            sample=sample,
            use_motion_veto=not args.no_motion_veto,
        )
        stage.on_reject = lambda frame, report: loop.on_rejected_frame()

    background: BackgroundStage | None = None
    if args.background or args.closed_loop:
        background = BackgroundStage(
            alpha=args.alpha, threshold=args.bg_threshold, force_backend=args.backend,
            downstream=loop,
        )
        stage.downstream = lambda frame, report: background(frame, report)
    pipeline = Pipeline(
        source,
        handler=stage,
        queue_size=args.queue_size,
        drop_when_full=not args.no_drop,
    )

    print(describe())
    print(f"running {args.source} source, queue depth {args.queue_size}, "
          f"drop_when_full={not args.no_drop}")
    metrics = pipeline.run()

    print(metrics.summary())
    print(f"gate: accepted={stage.accepted} rejected={stage.rejected} {stage.reasons or ''}")
    if background is not None:
        print(f"background: backend={background.backend} frames={background.frames} "
              f"mean foreground={background.mean_foreground_fraction:.2%}")
    if loop is not None:
        mean_stimulated = loop.stimulated_total / loop.cycles if loop.cycles else 0.0
        print(f"loop: segmenter={loop.backend} cycles={loop.cycles} "
              f"mean stimulated={mean_stimulated:.0f} px/frame")
        print(f"latency: {loop.latency_summary()}")
        if loop.sample is not None:
            print(f"sample: peak suppression={source.suppression.max():.2f} "
                  f"mean={source.suppression.mean():.3f}")
        else:
            print("sample: feedback not connected (open loop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
