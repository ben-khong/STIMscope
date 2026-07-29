# closed-loop-pipeline

A real-time imaging pipeline: acquire frames, reject the unusable ones, find where the
activity is, and steer a light-patterning device at it — then watch the sample respond on
the next frame.

```
camera ──▶ [ingest thread] ──queue──▶ [process thread] ──▶ gate ──▶ background ──▶ segment ──▶ DMD
   ▲                                                                                          │
   └──────────────────────── sample responds to stimulation ◀─────────────────────────────────┘
```

Python owns acquisition, gating and orchestration. C++ owns the per-frame pixel work. The
two are joined in-process through pybind11 — a frame that passes the gate is handed to a
C++ function as a view onto the same buffer, with no copy, no socket and no second
process. Background subtraction has three interchangeable backends: a NumPy reference, a
scalar C++ loop, and a hand-written CUDA kernel.

The repo runs end to end with no hardware attached: `SyntheticSource` stands in for the
camera, and every native path has a pure-Python fallback, so a fresh clone with nothing
compiled still passes its tests.

## Status

Parts 1 and 2 are in — ingest, diagnostic gating, the threaded pipeline, the native
extension and the CUDA background-subtraction kernel. Segmentation and DMD actuation are
part 3. See the roadmap.

## Install

```bash
python -m pip install -e ".[dev]"      # add ",camera" for live OpenCV capture
pytest -q                              # 42 passed, 2 skipped (CUDA tests) without a build
```

### Building the native extension

```bash
./scripts/build_native.sh
```

CUDA is auto-detected. On a Jetson (or any box with the toolkit) CMake finds `nvcc` and
builds the GPU backend; on a laptop it prints one line and builds the CPU-only extension.
Nothing downstream has to care — the module reports its backend at runtime and the Python
layer adapts. Force it either way with `-DCLPIPE_WITH_CUDA=ON` / `=OFF`.

Without a build, `clpipe` falls back to the NumPy implementation and says so:

```
native extension not built (...); using the NumPy reference.
```

## Run

```bash
clpipe --frames 300 --fps 30                    # paced like a 30fps camera
clpipe --frames 2000 --free-run --background    # unpaced, with background subtraction
clpipe --source camera --device 0 --background
python bench/bench_background.py --size 512 --frames 200
```

## Design notes

**Why two threads.** Acquisition is I/O-bound (waiting on the sensor), processing is
compute-bound. In lockstep a cycle costs `acquire + process`; overlapped it costs
`max(acquire, process)`. `tests/test_pipeline.py` asserts this empirically rather than
taking it on faith — a 5 ms source paired with a 5 ms handler must finish well under
10 ms per frame.

**Why the queue is bounded, and small.** An unbounded queue does not remove backpressure,
it converts backpressure into unbounded latency and memory growth. When the consumer falls
behind there are only two honest options: block the producer, or drop frames.

**Why dropping the *oldest* is the right default.** This pipeline acts on the current state
of the sample. A frame that has sat in a queue describes a state that no longer holds, and
acting on it steers the hardware at the past. So the producer never waits to enqueue: if
the queue is full it discards the stale head and takes the live frame. `--no-drop` flips to
blocking semantics, which is what you want when recording to disk.

**Why the gate is cheap.** It runs on every frame inside the real-time budget. Grayscale
plus a mean catches the two failure modes that actually occur on a rig — shutter closed
(mean below 10) and illumination spike or gain runaway (mean above 245) — for about a tenth
of a millisecond at 256×256. Anything more elaborate is taken out of the inference budget.

**Why the boundary is free.** The frame is already in a numpy buffer. `bindings.cpp` wraps
its pointer, shape and row stride in a `View2D` and calls straight into C++. Two things
enforce that rather than claiming it: every array argument is `.noconvert()`, so a dtype or
layout that would need a silent copy raises instead; and `probe_pointer` runs an array
through the identical conversion path so the test suite can assert
`native.probe_pointer(a) == a.ctypes.data`. The GIL is released for the duration of the
compute, which is what lets the ingest thread keep pulling frames.

**Why background subtraction, specifically, went to the GPU.** Each pixel's update depends
only on that pixel's own history — no neighbourhood term, no reduction, no sequential
dependency. That independence is the whole argument for one thread per pixel. The only
shared quantity is the foreground count, which the kernel reduces within each block so it
performs one atomic per block rather than one per foreground pixel.

**Why the GPU model stays resident.** The background model is a float per pixel. Shipping
it up and down every frame would cost more than the arithmetic it feeds and would turn the
benchmark into a measurement of the memory bus. Device buffers are allocated once at
construction and reused.

## Benchmark

Two numbers matter and they are not the same number. `bench/bench_background.py` prints
both. Measured on an x86 container with no GPU, 512×512:

```
stage only (background subtraction)
backend  median ms  p95 ms  best ms  vs numpy
numpy    0.614      0.646   0.579    1.00x
cpp      0.258      0.287   0.224    2.38x

end to end (source -> gate -> background)
backend  fps    latency p50 ms  latency p95 ms  vs numpy
numpy    128.7  0.94            4.51             +0.0%
cpp      138.1  0.52            3.94             +7.3%

cpp: 2.4x on the stage, +7.3% end to end. The gap is the rest of the loop.
```

That gap is Amdahl's law and it is the point of printing both. A stage that drops from
15 ms to 2 ms inside a 130 ms loop is a 7× stage win and roughly a 10% throughput win.
Quoting the first without the second is a claim that collapses under one follow-up
question. Rerun on the target board to get numbers that describe your hardware.

## Roadmap

| # | Commit | What lands |
|---|--------|-----------|
| 1 | `feat: frame sources` | `FrameSource` protocol, synthetic generator, OpenCV camera ✅ |
| 2 | `feat: diagnostic gate` | grayscale mean-intensity rejection + tests ✅ |
| 3 | `feat: threaded pipeline` | bounded queue, drop policy, metrics, CLI ✅ |
| 4 | `build: c++ extension` | CMake, pybind11, zero-copy views, scalar CPU backend ✅ |
| 5 | `perf: cuda kernel` | hand-written `__global__` kernel, block reduction, auto-detect ✅ |
| 6 | `feat: background stage` | wired into the pipeline behind the gate ✅ |
| 7 | `bench: throughput harness` | per-stage and end-to-end numbers ✅ |
| 8 | `feat: segmentation inference` | ONNX Runtime in C++, stub model so it runs bare |
| 9 | `feat: dmd actuation` | device abstraction + simulated sample response |
| 10 | `feat: close the loop` | full cycle wired, latency budget in README |

## Two claims worth getting right before you talk about this

**"Custom CUDA kernels."** Calling OpenCV's `cv::cuda` module is using someone else's
kernels, and the first question an interviewer asks is "show me the kernel."
`cpp/src/background_cuda.cu` is a real one — grid-stride indexing, a shared-memory tree
reduction for the count, persistent device state, and a documented reason why it is not
bit-identical to the CPU path (nvcc contracts the model update into an FMA, so parity is
asserted to within a couple of pixels rather than exactly). If you would rather describe
the original work precisely, the accurate phrasing is *"moved background subtraction to the
GPU via CUDA, cutting per-frame cost from ~15 ms to ~2 ms."*

**"Distributed."** Two threads sharing a `queue.Queue` in one process is concurrency, not
distribution. *"Concurrent"* or *"multi-stage real-time pipeline"* is accurate and is not a
weaker claim — the overlap argument above is the interesting part either way.
