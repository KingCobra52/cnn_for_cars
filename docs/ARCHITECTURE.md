# Architecture

## The decision everything else follows from

The project had four CPU cores and no GPU. That is not a footnote; it determined the
design.

The obvious approach to Stanford Cars is to fine-tune a pretrained backbone end to end.
On this hardware that is not available. A single epoch over 16k images at 224×224 through
a ViT takes roughly 40 minutes of forward pass alone, and backward roughly triples it. A
20-epoch run would be most of a day, and one run is not an experiment.

The way out is to notice that fine-tuning repeats work it does not have to. If the
backbone is frozen, its output for a given image is a constant. Computing it once per
epoch is computing the same number twenty times.

So the pipeline splits at that seam:

```
┌────────────────────────────────────────────────────────┐
│  Stage 1 — once per backbone, ~40 min                  │
│                                                         │
│  images ──► preprocess ──► frozen backbone ──► (N, D)  │
│                                                  │      │
│                                     embeddings.npy      │
└──────────────────────────────────────────────────┼──────┘
                                                   │
┌──────────────────────────────────────────────────┼──────┐
│  Stage 2 — 30 times, ~3 s each                   ▼      │
│                                                         │
│  embeddings ──► head ──► logits ──► metrics, CIs, ECE  │
└─────────────────────────────────────────────────────────┘
```

Stage 1 is paid three times in total — once per backbone. Stage 2 is where all the
experimentation happens, and it is nearly free.

## What that buys

A full sweep is 3 backbones × 2 head types × 5 seeds. Thirty runs. On cached embeddings
that finishes in minutes, which changes what is worth reporting:

- **Five seeds per configuration**, so run-to-run variance is measured rather than
  assumed away.
- **Bootstrap confidence intervals** on every headline number, so "84.9 vs 85.4" can be
  correctly reported as unresolved.
- **Calibration analysis**, which needs a held-out validation pass that would otherwise
  compete for the same scarce compute.

None of these are expensive ideas. They are usually skipped because each one costs
another training run, and here another training run costs three seconds.

## Tradeoffs this makes, honestly

**No augmentation on the training images.** Augmentation would change the input to the
backbone, so its embedding would differ per epoch, so the cache would be useless. This
costs accuracy — probably a couple of points. It is the price of the design, and the
right way to spend a GPU, if one arrives, is to lift exactly this restriction.

**Accuracy below a fine-tuned model.** A fully fine-tuned backbone on Stanford Cars beats
a linear probe on frozen features, and by a wide margin at the top of the leaderboard.
This project does not claim otherwise. What it claims is that a linear probe on strong
self-supervised features is a genuinely good model for the compute available, and that
measuring it carefully is worth more than measuring a better model carelessly.

**The backbone is a dependency, not a contribution.** The representation is DINOv2's or
CLIP's. The work here is the pipeline around it, the evaluation, and the analysis.

## Cache correctness

The cache is the one place where a bug would be both silent and total: serving stale
embeddings would corrupt every downstream number while every test still passed.

So the cache is content-addressed. The key is a SHA256 over:

- the backbone name,
- its exact pretrained weights tag (`IMAGENET1K_V1` and `V2` are different models),
- the full preprocessing spec — resize, crop, mean, std, interpolation,
- the ordered list of image ids **and their labels**,
- a fingerprint of the downloaded dataset, taken from `download.json`.

Change any of those and the key changes, so the entry is rebuilt rather than reused.

The last two were added after a review found the key could not do the job this section
claims. Image ids are positional (`train_00000`), so a key over ids alone reduced to
*(backbone, preprocessing, image count)*. Swapping the Hub mirror for another with the
same number of images produced an identical key, and the cache then served the previous
mirror's embeddings against the new mirror's labels — silently, corrupting every
downstream number with nothing failing. Labels and the dataset fingerprint close that.

`tests/test_cache_key.py` checks each input individually, including that ids are
separated when hashed — without a separator, `["ab", "c"]` and `["a", "bc"]` would
collide.

Builds are resumable. Embeddings go to a memory-mapped array flushed every 512 rows with
progress recorded alongside, because an hour-long build should survive a closed laptop.
An interrupted entry has a `progress.json` and no `manifest.json`, and `is_complete`
treats that as incomplete — so a partial build is never mistaken for a finished one.

## Where preprocessing lives

Preprocessing belongs to the backbone, not to the caller. `BackboneSpec` binds a
`PreprocessSpec` to the weights it was chosen for, and nothing constructs a transform by
hand.

This is a structural fix for a real bug in the project this replaced, which fed 32×32
CIFAR-normalised tensors to a pretrained ImageNet VGG16. That mistake was invisible
because the task was easy enough that the model still scored 85%. Making the two
inseparable means it cannot recur, and `PreprocessSpec` being hashable means a change to
it invalidates the cache.

Binding the two together is necessary but not sufficient: the constants still have to be
*right*. `resnet50` was bound to a 256/bicubic transform when its IMAGENET1K_V2 weights
want 232/bilinear — a quieter version of the same mistake, degrading every embedding
that backbone produced. `tests/test_transforms.py` now asserts each torchvision
backbone's spec against `Weights.transforms()` directly, so the claim is checked rather
than asserted. `app/serving.py` reads the interpolation from the bundle for the same
reason; it previously hardcoded bicubic, which was invisible while every backbone
happened to use it.

## Evaluation discipline

`eval.py` is the only module that reads the test set. Everything that involves a choice —
early stopping, best-run selection, the calibration temperature — reads validation.

This is not a stylistic preference. The project this replaced had no validation split at
all: the test set was the only held-out data, and it was used both for tuning and for the
headline number, which makes that number meaningless in a way no amount of care
elsewhere can repair.

## Serving

Training reads embeddings; serving must take an image. So `export.py` fuses backbone,
head and calibration temperature into one `ServingModel` and exports a single ONNX graph
from preprocessed image to calibrated logits.

Every export is verified against PyTorch on a probe batch before it is reported as
successful. Operator coverage differs between the two runtimes and a silently wrong
export is a realistic failure; `tests/test_export.py` checks that `verify()` actually
rejects a mismatched model, so the check cannot quietly become inert.

The serving image does not install torch. The ONNX graph carries the model, so the
runtime needs only `onnxruntime` and PIL — which is the difference between a ~150 MB
image and a ~2 GB one, and it is why the demo fits comfortably on a free CPU Space.
