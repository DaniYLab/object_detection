# ArchCAD batched step — infrastructure measurement 2026-09-17

## Why

Profiling one frozen unbatched training step (149,343 parameters, 147 primitives,
8 segments) gives 10.5 ms of wall clock on the RTX 5070 Ti for ~69 MFLOP of
arithmetic, i.e. ~1.7 us of compute or **0.016% of the step**. The forward is barely
faster on CUDA than on CPU (2.137 ms vs 2.800 ms; full step ratio 0.79x). The frozen
pilot is host/launch bound, so the GPU cannot be used harder without changing how
much work each launch carries.

This document records an **infrastructure measurement**, not a research result. It
changes no scientific claim and produces no benchmark numbers of its own.

## What was built

`src/models/archcad_primitive_batched.py` and
`scripts/research/train_archcad_primitive_batched.py`. The batched model subclasses
the frozen one and only overrides `forward`, so `state_dict` keys are identical and
weights transfer in either direction; existing checkpoints and diagnostics work
unchanged. Padding is right-aligned and every consumer slices `[:, :n]`, so the
frozen decoder and evaluator stay on the reporting path.

## Equivalence checks (all passed)

| Check | Result |
|---|---|
| `state_dict` keys batched vs frozen | identical |
| B=1 forward, class and mask logits | **bit-exact**, max delta `0.000e+00` |
| per-image loss, padded batch vs unbatched (B=1/2/4) | max abs delta 1.2e-07 – 1.7e-06 |
| full evaluation metrics, B=1/2/4/8, 16 val images | **identical** to the frozen evaluator |
| batch of real images with 72% padding | short image unaffected by pad positions |
| 7 unit tests, `tests/test_archcad_batched_step.py` | pass |

The residual loss difference is float32 reassociation from passing an all-False
padding mask, which selects a different attention kernel; it does not reach the
metrics.

## Measured throughput, 64 real training images

| Path | images/s | vs frozen | padding |
|---|---:|---:|---:|
| frozen unbatched | 106.7 | 1.00x | — |
| B=4 shuffled | 201.8 | 1.89x | 50.4% |
| B=8 shuffled | 239.4 | 2.24x | 61.4% |
| B=4 length-grouped | 215.0 | 2.02x | 7.6% |
| B=8 length-grouped | 271.3 | 2.54x | 16.2% |
| **B=16 length-grouped** | **309.3** | **2.90x** | 31.1% |

Padding is nearly free because compute is nearly free: B=16 with 72% padding still
gives 2.61x, and grouping by primitive count adds the rest. Grouping is the better
default, so it is the trainer default.

## Combined with process concurrency

Both levers attack a host-side bottleneck, so they were measured together rather
than assumed to multiply. 200 epochs x 8 images, batched B=8, length-grouped:

| K concurrent runs | wall | runs/min | vs one batched run |
|---:|---:|---:|---:|
| 1 | 14.3 s | 4.19 | 1.00x |
| 4 | 25.2 s | 9.51 | 2.27x |
| 8 | 42.8 s | 11.21 | 2.68x |

One batched run alone took 14.3 s against 31.3 s for the frozen unbatched path, so
batching (~2.2x) and concurrency (~2.7x at K=8) combine to roughly **5.9x** total
throughput on this workload. Concurrency alone, measured separately on the frozen
path, saturated at 2.3x.

## Using it for results

This is a different optimization protocol: one optimizer step now covers a batch of
images, so steps per epoch, the effective gradient batch and the loss normalization
all change. Comparisons against the frozen per-image protocol must be preregistered
as such; nothing here licenses replacing the frozen baseline's numbers.

The value of the speedup is in the next phase, which is ablation-heavy: raster
feature sampling and fusion arms, each needing several seeds.
