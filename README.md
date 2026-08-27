# closed-loop-pipeline

A real-time imaging pipeline: acquire frames, reject the unusable ones, find where the
activity is, steer a light-patterning device at it — then watch the sample respond on the
next frame and do it again.

```
camera ─▶ [ingest thread] ─queue─▶ [process thread] ─▶ gate ─▶ background ─▶ segment ─▶ DMD
   ▲                                                                                     │
   └───────────────────── sample responds to stimulation ◀────────────────────────────────┘
```

Python owns acquisition, gating and orchestration. C++ owns the per-frame pixel work:
background subtraction, segmentation, and the actuation path. The two are joined
in-process through pybind11 — a frame that passes the gate is handed to a C++ function as
a view onto the same buffer, with no copy, no socket and no second process.

Three things are pluggable, and all three degrade rather than fail: background subtraction
runs on a hand-written CUDA kernel, a scalar C++ loop, or NumPy; segmentation runs an ONNX
model or a built-in analytic operator; the DMD is a real device or a simulated one. The
repo runs end to end with nothing compiled and no hardware attached, because the simulated
sample closes the loop in software.

## Status

All three parts are in. Nothing below is a stub except the things that have to be: the
trained model and the vendor DMD SDK, both of which are behind interfaces with working
stand-ins.

## Install

```bash
python -m pip install -e ".[dev]"      # add ",camera" for live OpenCV capture
pytest -q                              # passes with nothing compiled
```

### Building the native extension

```bash
./scripts/build_native.sh
```

Two optional dependencies, both auto-negotiated:

**CUDA** is detected. On a Jetson CMake finds `nvcc` and builds the GPU background-
subtraction kernel; on a laptop it prints one line and builds CPU-only. Force with
`-DCLPIPE_WITH_CUDA=ON|OFF`.

**ONNX Runtime** has to be pointed at, since its C++ SDK has no reliable install location:

```bash
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf ort.tgz
./scripts/build_native.sh -DCLPIPE_ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.20.1
```

Without it the extension builds with the analytic segmenter only, and asking for a `.onnx`
model at runtime raises instead of silently substituting something else.

### A model to run

The trained model this was built around belongs to the lab. `scripts/make_stub_model.py`
emits a shape-compatible stand-in — NCHW float in and out, dynamic spatial dims, computing
centre-surround contrast — so the ONNX path can actually be exercised:

```bash
python scripts/make_stub_model.py --output models/stub_segmenter.onnx
```

Swapping in the real model needs no code change, only a different path.

## Run

```bash
clpipe --frames 300 --fps 30                              # ingest + gate
clpipe --frames 2000 --free-run --background              # + background subtraction
clpipe --frames 300 --free-run --closed-loop              # the whole loop
clpipe --frames 300 --free-run --closed-loop --open-loop  # same, feedback disconnected
clpipe --closed-loop --segmenter models/stub_segmenter.onnx
clpipe --source camera --device 0 --closed-loop
python bench/bench_background.py --size 512 --frames 200
```

## Is the loop actually closed?

The word gets used loosely, so the repo makes it falsifiable. `--open-loop` runs every
identical stage over identical frames and disconnects one edge: the projected pattern no
longer reaches the sample. Everything else is held constant.

```
closed:  cycles=283  mean stimulated=61 px/frame   peak suppression=0.69
open:    cycles=283  mean stimulated=80 px/frame   feedback not connected
```

Closed, the loop needs less stimulation over time, because the stimulation is working: lit
regions dim, drop below the activation threshold, stop being targeted, and recover. Open,
it keeps firing at full strength forever because nothing it does ever comes back. That gap
is the entire difference between a control loop and a detector, and
`tests/test_closedloop.py` asserts it rather than describing it.

## Design notes

**Why two threads.** Acquisition is I/O-bound, processing is compute-bound. In lockstep a
cycle costs `acquire + process`; overlapped it costs `max(acquire, process)`.
`tests/test_pipeline.py` asserts this empirically — a 5 ms source paired with a 5 ms
handler must finish well under 10 ms per frame.

**Why the queue is bounded and small, and why it drops the oldest.** An unbounded queue
does not remove backpressure, it converts backpressure into unbounded latency and memory
growth. This pipeline acts on the current state of the sample, so a frame that has sat in
a queue describes a state that no longer holds and steers the hardware at the past. The
producer therefore never waits to enqueue: if the queue is full it discards the stale head
and takes the live frame.

You can watch this engage. The ONNX stand-in costs about 10 ms per frame at 256×256, which
is slower than the free-running source, so the pipeline sheds frames and the loop keeps
running on fresh ones instead of accumulating a backlog. `--no-drop` flips to blocking
semantics, which is what you want when recording to disk.

**Why the gate is cheap.** It runs on every frame inside the real-time budget. Grayscale
plus a mean catches the two failure modes that actually occur on a rig — shutter closed
and illumination spike — for about a tenth of a millisecond at 256×256.

**Why a rejected frame blanks the mirrors.** Rejection is not a no-op downstream. A system
that cannot see the sample has no basis for illuminating it, and holding the last good
pattern on the device while blind is worse than going dark.

**Why the boundary is free.** `bindings.cpp` wraps the numpy buffer's pointer, shape and
row stride in a `View2D` and calls straight into C++. Two things enforce that rather than
claim it: every array argument is `.noconvert()`, so a dtype or layout needing a silent
copy raises instead; and `probe_pointer` runs an array through the identical conversion
path so the tests can assert `native.probe_pointer(a) == a.ctypes.data`. The GIL is
released around the compute, which is what lets the ingest thread keep pulling frames.
The DMD pattern crosses back the same way — a read-only view onto the device's own buffer,
with the controller as its base object.

**Why background subtraction went to the GPU.** Each pixel's update depends only on that
pixel's own history — no neighbourhood term, no reduction, no sequential dependency. That
independence is the argument for one thread per pixel. The only shared quantity is the
foreground count, which the kernel reduces within each block so it performs one atomic per
block rather than one per foreground pixel. Device buffers are allocated once, because
shipping a float-per-pixel model across the bus twice per frame would cost more than the
arithmetic it feeds.

**Why segmentation and not classification.** A classifier answers "is something happening"
with one label per frame. The DMD needs to know *where* to aim, so a per-frame label is
unusable no matter how accurate. A segmenter answers per pixel, and that map is what
steers the mirrors. `test_output_is_a_map_not_a_label` moves the stimulus across the field
and asserts the output moves with it.

**Why the motion mask is a veto.** A pixel is stimulated only where the segmenter says
active *and* background subtraction says something changed there. Two independent signals
have to agree before light goes anywhere near the sample.

**What deployment actually costs.** The ONNX backend creates its session, allocator, input
tensor and output buffer once at construction, pins intra-op threading to one thread, and
validates the model's input geometry against the camera at startup. Per-frame allocation
inside a loop with a deadline shows up as p99 latency long before it shows up in a mean;
ORT's default thread pool fills the machine and on a Jetson fights the acquisition thread
for cores, making latency worse under load; and a model whose spatial dims disagree with
the camera should fail at startup rather than halfway through an experiment.

## Benchmark

Two numbers matter and they are not the same number. `bench/bench_background.py` prints
both. Rerun it on your own hardware — the numbers in this section describe the machine
they were measured on and nothing else.

The stage number is how long a stage takes; the end-to-end number is what the loop
sustains. A stage rewrite only moves the second one if that stage was the constraint.
Measured on an M-series MacBook, CPU-only build, 512×512, background subtraction went
1.95× faster per stage and the loop's throughput did not move at all, because on that
machine the frame source costs more per frame than the processing does — the pipeline is
producer-bound, and speeding up a consumer that is already idle buys latency, not
throughput. That is Amdahl's law with a concrete face on it, and it is the honest answer
to "you said 2× faster, why is the pipeline the same speed."

The same applies to the segmenter, in the opposite direction. The analytic operator runs
in about 0.6 ms via an integral image, which is O(1) per pixel regardless of window size;
the 17×17 convolution in the ONNX stand-in costs roughly 10 ms for the same field. The
analytic version is algorithmically better *at this particular job*. A trained model earns
its cost when the task needs learned features rather than a hand-designed filter — which
is exactly why the engine is a swappable interface and not a hard-coded choice.

## Layout

```
clpipe/            Python: acquisition, gating, orchestration, NumPy references
cpp/include/       View2D, and the Segmenter / DmdDevice / controller interfaces
cpp/src/           CPU + CUDA background subtraction, segmenters, DMD, bindings
bench/             throughput harness
scripts/           build_native.sh, make_stub_model.py
tests/             69 tests; every native path has a skip-if-unbuilt guard
```

## Two claims worth getting right before you talk about this

**"Custom CUDA kernels."** Calling OpenCV's `cv::cuda` module is using someone else's
kernels, and the first question an interviewer asks is "show me the kernel."
`cpp/src/background_cuda.cu` is a real one — grid-stride indexing, a shared-memory tree
reduction for the count, persistent device state, and a documented reason why it is not
bit-identical to the CPU path. If you would rather describe the original work precisely,
the accurate phrasing is *"moved background subtraction to the GPU via CUDA, cutting
per-frame cost from ~15 ms to ~2 ms."*

**"Distributed."** Two threads sharing a `queue.Queue` in one process is concurrency, not
distribution. *"Concurrent"* or *"multi-stage real-time pipeline"* is accurate and is not a
weaker claim — the overlap argument above is the interesting part either way.
