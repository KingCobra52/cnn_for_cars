# Results

> **Not yet generated.** This file is written by `carvision report` from the run
> directories under `artifacts/runs/`, after `make all` has been run on a machine with the
> dataset. Until then every cell reads TBD.
>
> Nothing in this file is typed by hand, and that is the point — see
> [`LESSONS.md`](LESSONS.md) for why.

## Headline

Stanford Cars, official 8,041-image test split. Intervals are 95% bootstrap CIs over the
test set; `±` is the standard deviation across 5 training seeds.

| Backbone | Head | Top-1 | Top-5 | Macro-F1 | Seeds |
| --- | --- | --- | --- | --- | --- |
| — | CLIP zero-shot | TBD | TBD | TBD | n/a |
| resnet50 | linear | TBD | TBD | TBD | TBD |
| resnet50 | mlp | TBD | TBD | TBD | TBD |
| clip_vitb32 | linear | TBD | TBD | TBD | TBD |
| clip_vitb32 | mlp | TBD | TBD | TBD | TBD |
| dinov2_vits14 | linear | TBD | TBD | TBD | TBD |
| dinov2_vits14 | mlp | TBD | TBD | TBD | TBD |

## Pairwise comparisons

Paired bootstrap on the difference in top-1, run with `carvision compare`. A difference
is only claimed where the interval excludes zero.

| A | B | Difference | Resolved? |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |

## Calibration

| Model | ECE (raw) | ECE (scaled) | Temperature |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |

## Error structure

| Error kind | Count | Share of errors |
| --- | --- | --- |
| Same model, different year | TBD | TBD |
| Same model, different body style | TBD | TBD |
| Same make, different model | TBD | TBD |
| Different make entirely | TBD | TBD |

Make-level top-1 (only the manufacturer has to be right): TBD

Share of errors where the true class was still in the top 5: TBD

## Most confused pairs

| True | Predicted | Count |
| --- | --- | --- |
| TBD | TBD | TBD |

## Hardest classes

Lowest per-class recall.

| Class | Recall | Support |
| --- | --- | --- |
| TBD | TBD | TBD |

## Latency (CPU, batch 1)

| Runtime | p50 | p95 |
| --- | --- | --- |
| PyTorch | TBD | TBD |
| ONNX Runtime | TBD | TBD |

## Cache build cost

The one-time price of the design, per backbone, over all 16,185 images.

| Backbone | Embedding dim | Build time | Cache size |
| --- | --- | --- | --- |
| resnet50 | 2048 | TBD | TBD |
| clip_vitb32 | 512 | TBD | TBD |
| dinov2_vits14 | 384 | TBD | TBD |

Against that: a 30-run sweep on cached embeddings takes TBD.
