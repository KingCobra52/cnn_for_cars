# carvision

**Fine-grained car recognition — 196 classes, built without a GPU.**

[![CI](https://github.com/KingCobra52/cnn_for_cars/actions/workflows/ci.yml/badge.svg)](https://github.com/KingCobra52/cnn_for_cars/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
<!-- Added in the deploy step: [![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20demo-yellow)](SPACES_URL) -->

Identifies the make, model and year of a car from a photograph, across the 196 classes of
Stanford Cars — a dataset where distinguishing a 2012 from a 2007 of the same model comes
down to a bumper and a badge.

> Results below are generated from experiment artifacts. Strict publication checks their completeness and provenance.

---

## Constraints

This project has no GPU. Fine-tuning a backbone on 196 classes was therefore not
available — a single epoch would have taken hours, and a meaningful experiment needs
dozens.

The design follows from that. A **frozen** backbone's output for a given image never
changes, so running it over the dataset is not a per-epoch cost at all. It is a one-time
cost, and what it produces is a 16,185 × D matrix that fits in memory. Fitting a 196-way
classifier over that matrix avoids repeated backbone inference.

So the expensive part became a cache, and the cheap part became the experiment:

```
        images ──► frozen backbone ──► embeddings.npy    (once per backbone)
                                            │
                                            ▼
                                     classifier head      (30 runs)
```

That inversion is what makes the rest of this repository affordable: **3 backbones × 2
head types × 5 seeds = 30 runs**, which is what pays for the confidence
intervals, the seed-to-seed spread, and the calibration analysis below. Those are things
GPU-rich projects routinely skip, and here they came for free.

The cache is content-addressed on the backbone identity, the exact pretrained weights
tag, the full preprocessing spec, and the ordered image ids — so it is not possible to
train on embeddings computed with different preprocessing than the config claims.

<!-- carvision:results:start -->
## Results

Results pending. Run the complete pipeline to generate validated metrics and analysis.
<!-- carvision:results:end -->

## Reproducing

```bash
make setup                 # repository-local .venv + validated dependencies
make data                  # download Stanford Cars, write the deterministic splits
make cache-all             # embed with all three backbones — one pass per backbone
make sweep                 # 30 runs on cached embeddings
make eval figures          # metrics, CIs, calibration, and every figure
make export bench          # ONNX export with parity check, plus latency numbers
```

Or `make all` for the lot. `make check` runs what CI runs: ruff, mypy strict, pytest.

All commands use `.venv/bin` directly, so shell activation is not required. On Linux,
`make setup` installs the CPU build of PyTorch; on Apple Silicon macOS it uses the normal
PyPI wheels. Override either default explicitly with:
`make setup TORCH_INDEX=https://download.pytorch.org/whl/cu124`.

For the exact versions this was developed against, `make sync` reinstalls the pinned set in
`requirements.lock`. Setup uses the same lockfile.

The test suite needs neither the dataset nor a network — every test runs on synthetic
fixtures — so a fresh clone is green in seconds.

**Reproducibility.** `set_seed` is called at every entry point. The train/val/test splits
are committed as CSVs, so you evaluate on exactly the images this README reports on.
Every run writes its checkpoint, fully resolved config, metrics and git SHA.

## Layout

```
src/carvision/
├── data/          download, deterministic splits, datasets, per-backbone preprocessing
├── features/      cache.py — the embedding cache this whole design rests on
├── models/        frozen backbone registry, heads, CLIP zero-shot, temperature scaling
├── metrics/       classification, bootstrap CIs, calibration
├── interpret/     Grad-CAM, error analysis by make / model / body style
├── train.py       head training on cached embeddings
├── eval.py        the only module that touches the test set
├── export.py      ONNX export + parity verification + latency benchmark
└── figures.py     every figure, generated from evaluation output
app/               Gradio demo (ONNX Runtime, no torch)
configs/           dataset acquisition (the mirror, columns, split seed)
notebooks/legacy/  the project this replaced, and what was wrong with it
```

## Where this came from

This repository previously held a single Colab notebook that classified CIFAR-10 into
`background` / `car` / `truck` at 32×32 and reported ~83% and ~85.5%.

It is preserved in [`notebooks/legacy/`](notebooks/legacy/README.md) with a writeup of
the six classes of defect found in its committed state — among them a figure plotting a
variable that was defined nowhere (it rendered only because of stale kernel state), loss
curves that were running totals rather than per-epoch means, a pretrained ImageNet model
fed CIFAR-normalised 32×32 input, and no seeds anywhere.

[`docs/LESSONS.md`](docs/LESSONS.md) maps each of those to the thing in this codebase
that makes it structurally impossible to repeat. That mapping is most of why the
architecture looks the way it does.

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the cache-first design and its tradeoffs
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — intended use, limitations, ethical notes
- [`docs/RESULTS.md`](docs/RESULTS.md) — the full run table
- [`docs/LESSONS.md`](docs/LESSONS.md) — before and after

## License

MIT. Stanford Cars is distributed for research use; see the model card.

## Publication workflow

Individual targets run only their named stage; `make report` never retrains a model.
`make all` runs stages sequentially, including comparisons and benchmarks, before strict
publication. Use `carvision report` for an explicitly incomplete/unverified draft, or
`make report` to require the complete active sweep and current artifacts.

The report generator updates only the marked README section and `docs/RESULTS.md`.
Stale artifact errors identify the producing command to rerun. Older artifacts without
versioned provenance must be regenerated for strict publication. Interrupted split updates
are recovered with `carvision data split --overwrite`.
