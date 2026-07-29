#!/usr/bin/env python3
"""Background-subtraction throughput, per stage and end to end.

Two numbers matter and they are not the same number.

The *stage* number is how long background subtraction itself takes. That is
where a GPU rewrite pays off, and it is the number that looks impressive.

The *end-to-end* number is how many frames per second the whole loop sustains.
Acquisition, the diagnostic gate and everything downstream are unchanged by the
rewrite, so by Amdahl's law the end-to-end gain is bounded by how much of the
budget the stage held in the first place. A stage that drops from 15 ms to 2 ms
inside a 130 ms loop is a 7x stage speedup and roughly a 10% throughput gain.

Both are printed, because quoting the first without the second is the kind of
claim that falls apart under one follow-up question.

    python bench/bench_background.py --size 512 --frames 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import List, Optional

import numpy as np

from clpipe import Pipeline, QualityStage, SyntheticSource
from clpipe.background import BackgroundStage, make_subtractor
from clpipe.native import HAVE_NATIVE, describe, native


def available_backends() -> List[str]:
    backends = ["numpy"]
    if HAVE_NATIVE:
        backends.append("cpp")
        if native.cuda_available():
            backends.append("cuda")
    return backends


def make_stream(size: int, count: int, seed: int = 0) -> List[np.ndarray]:
    """Pre-generate frames so generation cost stays out of the stage timing."""
    source = SyntheticSource(
        width=size, height=size, n_frames=count, realtime=False,
        blank_rate=0.0, overexposed_rate=0.0, seed=seed,
    )
    return [np.ascontiguousarray(frame.data) for frame in source]


def time_stage(backend: str, stream: List[np.ndarray], alpha: float, threshold: int) -> dict:
    size = stream[0].shape[0]
    subtractor = make_subtractor(size, size, alpha, threshold, force_backend=backend)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Warm up: seeds the model, faults in the buffers, and on the GPU pays the
    # one-off context-creation cost that would otherwise land on frame one.
    for frame in stream[:5]:
        subtractor.apply(frame, mask)

    samples = []
    for frame in stream:
        start = time.perf_counter()
        subtractor.apply(frame, mask)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "backend": backend,
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "min_ms": samples[0],
    }


def time_end_to_end(backend: str, size: int, frames: int, alpha: float, threshold: int) -> dict:
    """Full loop, unpaced: source -> gate -> background stage."""
    stage = BackgroundStage(alpha=alpha, threshold=threshold, force_backend=backend)
    gate = QualityStage(downstream=lambda frame, report: stage(frame, report))
    source = SyntheticSource(width=size, height=size, n_frames=frames, realtime=False, seed=1)

    metrics = Pipeline(source, handler=gate, queue_size=8).run()
    return {
        "backend": backend,
        "fps": metrics.fps,
        "latency_p50_ms": metrics.latency_ms(50),
        "latency_p95_ms": metrics.latency_ms(95),
    }


def print_table(rows: List[dict], columns: List[tuple]) -> None:
    widths = [max(len(header), *(len(fmt(row)) for row in rows)) for header, fmt in columns]
    print("  ".join(h.ljust(w) for (h, _), w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(fmt(row).ljust(w) for (_, fmt), w in zip(columns, widths)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=512, help="frame edge length in pixels")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--backends", nargs="*", default=None,
                        help="subset of: numpy cpp cuda (default: all available)")
    parser.add_argument("--skip-end-to-end", action="store_true")
    args = parser.parse_args(argv)

    backends = args.backends or available_backends()
    unavailable = set(backends) - set(available_backends())
    if unavailable:
        print(f"unavailable backend(s): {', '.join(sorted(unavailable))}", file=sys.stderr)
        print(describe(), file=sys.stderr)
        return 1

    print(describe())
    print(f"{args.size}x{args.size} frames, {args.frames} per backend, "
          f"alpha={args.alpha} threshold={args.threshold}\n")

    stream = make_stream(args.size, args.frames)

    print("stage only (background subtraction)")
    stage_rows = [time_stage(b, stream, args.alpha, args.threshold) for b in backends]
    print_table(stage_rows, [
        ("backend", lambda r: r["backend"]),
        ("median ms", lambda r: f"{r['median_ms']:.3f}"),
        ("p95 ms", lambda r: f"{r['p95_ms']:.3f}"),
        ("best ms", lambda r: f"{r['min_ms']:.3f}"),
        ("vs numpy", lambda r: f"{stage_rows[0]['median_ms'] / r['median_ms']:.2f}x"),
    ])

    if args.skip_end_to_end:
        return 0

    print("\nend to end (source -> gate -> background)")
    loop_rows = [time_end_to_end(b, args.size, args.frames, args.alpha, args.threshold)
                 for b in backends]
    print_table(loop_rows, [
        ("backend", lambda r: r["backend"]),
        ("fps", lambda r: f"{r['fps']:.1f}"),
        ("latency p50 ms", lambda r: f"{r['latency_p50_ms']:.2f}"),
        ("latency p95 ms", lambda r: f"{r['latency_p95_ms']:.2f}"),
        ("vs numpy", lambda r: f"{r['fps'] / loop_rows[0]['fps'] - 1.0:+.1%}"),
    ])

    if len(stage_rows) > 1:
        fastest = min(stage_rows, key=lambda r: r["median_ms"])
        loop = next(r for r in loop_rows if r["backend"] == fastest["backend"])
        stage_gain = stage_rows[0]["median_ms"] / fastest["median_ms"]
        loop_gain = loop["fps"] / loop_rows[0]["fps"] - 1.0
        print(f"\n{fastest['backend']}: {stage_gain:.1f}x on the stage, "
              f"{loop_gain:+.1%} end to end. The gap is the rest of the loop.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
