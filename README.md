# closed-loop-pipeline

A real-time imaging pipeline: acquire frames, reject the unusable ones, segment the
usable ones, and steer a light-patterning device at whatever the segmentation found —
then watch the sample respond on the next frame.

```
camera ──▶ [ingest thread] ──queue──▶ [process thread] ──▶ gate ──▶ segment ──▶ DMD
   ▲                                                                            │
   └──────────────────── sample responds to stimulation ◀───────────────────────┘
```

Python owns acquisition, gating and orchestration. C++ owns the per-frame inference
and actuation path. The two are joined in-process through pybind11 — a frame that
passes the gate is handed to a C++ function as a view onto the same buffer, with no
copy, no socket and no second process.

The repo runs end to end with no hardware attached: `SyntheticSource` stands in for the
camera and a simulated sample closes the loop, so the concurrency, the timing and the
feedback behaviour are all observable on a laptop.

## Status

Part 1 of 3 is in: ingest, the diagnostic gate, and the threaded pipeline that connects
them. See the roadmap below.

## Install

```bash
python -m pip install -e ".[dev]"      # add ",camera" for live OpenCV capture
pytest -q
```

## Run

```bash
clpipe --frames 300 --fps 30           # synthetic source, paced like a 30fps camera
clpipe --frames 2000 --free-run        # unpaced: measures the processing ceiling
clpipe --source camera --device 0      # real capture
clpipe --frames 300 --queue-size 2 --verbose
```

Output:

```
captured=300 processed=300 dropped=0 (0.0%) throughput=30.1 fps  latency p50=0.11ms p95=0.16ms
gate: accepted=288 rejected=12 {'blank': 7, 'overexposed': 5}
```

## Design notes

**Why two threads.** Acquisition is I/O-bound (waiting on the sensor) and processing is
compute-bound. Running them in lockstep costs `acquire + process` per cycle; overlapping
them costs `max(acquire, process)`. `tests/test_pipeline.py` asserts this empirically
rather than taking it on faith — a 5 ms source paired with a 5 ms handler must finish
well under 10 ms per frame.

**Why the queue is bounded, and small.** An unbounded queue does not remove backpressure,
it converts backpressure into unbounded latency and memory growth. When the consumer
falls behind there are only two honest options: block the producer, or drop frames.

**Why dropping the *oldest* is the right default.** This pipeline exists to act on the
current state of the sample. A frame that has been sitting in a queue describes a state
that no longer holds, and acting on it steers the hardware at the past. So the producer
never waits to enqueue: if the queue is full it discards the stale head and enqueues the
live frame. `--no-drop` flips to blocking semantics, which is what you want when
recording to disk and no frame may be lost.

**Why the gate is so cheap.** It runs on every frame inside the real-time budget.
Converting to grayscale and taking a mean catches the two failure modes that actually
occur on a rig — shutter closed (mean below 10) and illumination spike or gain runaway
(mean above 245) — for roughly a tenth of a millisecond at 256×256. Anything more
elaborate here is taken straight out of the inference stage's budget.

## Roadmap

| # | Commit | What lands |
|---|--------|-----------|
| 1 | `feat: frame sources` | `FrameSource` protocol, synthetic generator, OpenCV camera ✅ |
| 2 | `feat: diagnostic gate` | grayscale mean-intensity rejection + tests ✅ |
| 3 | `feat: threaded pipeline` | bounded queue, drop policy, metrics, CLI ✅ |
| 4 | `build: c++ core + pybind11` | CMake, zero-copy `ndarray` → `cv::Mat`, round-trip test |
| 5 | `feat: cpu background subtraction` | the naive per-pixel loop — the "before" number |
| 6 | `perf: cuda background kernel` | hand-written `__global__` kernel + CPU fallback |
| 7 | `bench: throughput harness` | per-stage and end-to-end numbers, reproducible |
| 8 | `feat: segmentation inference` | ONNX Runtime in C++, stub model so it runs bare |
| 9 | `feat: dmd actuation` | device abstraction + simulated sample response |
| 10 | `feat: close the loop` | full cycle wired, metrics in README |

## Two claims worth getting right before you talk about this

**"Custom CUDA kernels."** Calling OpenCV's `cv::cuda` module is using someone else's
kernels, and the first question an interviewer asks is "show me the kernel." Step 6
writes a real `__global__` function so the sentence is backed by the repo. If you would
rather the résumé match the original work exactly, the honest phrasing is *"moved
background subtraction to the GPU via CUDA, cutting per-frame cost from ~15 ms to
~2 ms."*

**"Distributed."** Two threads sharing a `queue.Queue` in one process is concurrency, not
distribution. *"Concurrent"* or *"multi-stage real-time pipeline"* is the accurate word
and is not a weaker claim — the overlap argument above is the interesting part either way.

**On the 10%.** A stage going 15 ms → 2 ms is a ~7× speedup of that stage but only ~10%
end to end, because the rest of the loop dominates. Step 7's harness reports both, so the
number is defensible when someone asks why 7× at the stage became 10% overall.
