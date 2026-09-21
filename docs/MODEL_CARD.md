# Model card: carvision

## Overview

A fine-grained car classifier over the 196 classes of Stanford Cars. A frozen pretrained
backbone produces image embeddings; a small classifier head trained on those embeddings
produces class probabilities, temperature-calibrated on a held-out validation split.

- **Task:** single-label classification, 196 classes of make / model / year / body style
- **Input:** one RGB photograph
- **Output:** a probability distribution over the 196 classes
- **Training:** the head only. The backbone is frozen and never updated.

## Intended use

Built as a portfolio and educational project demonstrating a complete, reproducible
computer-vision pipeline under a CPU-only constraint.

**Appropriate:** exploring transfer learning and linear probing; a reference for
cache-first ML pipeline design; casual identification of car models in photographs.

**Not appropriate**, and this list is meant literally:

- **Any surveillance or tracking application.** Vehicle recognition is a surveillance
  technology. This model identifies makes and models, not individual vehicles or people,
  but it is a component of systems that do, and it is not offered for that use.
- **Law enforcement, insurance, or any consequential decision.** The model is wrong a
  meaningful fraction of the time and its errors are not uniformly distributed.
- **Vehicle valuation, damage assessment, or transactions.** It cannot see condition,
  mileage, trim, or damage.
- **Safety-critical systems.** No robustness or adversarial testing has been done.

## Training data

Stanford Cars: 16,185 images across 196 classes, split 8,144 train / 8,041 test. The
official test split is untouched. A seeded, stratified 15% of the official train split is
held out for validation; the remainder is used for training.

The dataset is distributed for research use. The canonical Stanford AI Lab download is
offline, so images are obtained from a Hugging Face Hub mirror; the exact repository and
revision used are recorded in `data/stanford_cars/download.json`.

**Known properties of this data, which become properties of the model:**

- Overwhelmingly **US-market vehicles from roughly 1991–2012**. Performance on newer
  cars, and on models sold primarily in Europe or Asia, is untested and should be
  expected to be poor.
- Photographs are mostly **clear, daylight, unoccluded, side or three-quarter views**,
  many of them promotional or enthusiast shots. Night, rain, heavy occlusion, unusual
  angles and low resolution are all out of distribution.
- Roughly **40 training images per class**. The tail classes are learned from very
  little.

The downloaded mirror's content audit records 39 exact-content duplicate groups,
including 19 spanning the official training and test partitions. The official partition
is preserved for comparability; disjoint image IDs do not guarantee unique image content.

## Evaluation

Reported on the official 8,041-image test split, held out from model selection. See
[`RESULTS.md`](RESULTS.md) for the numbers and the confidence intervals.

Metrics: top-1 and top-5 accuracy, macro-F1, balanced accuracy, per-class recall,
expected calibration error before and after temperature scaling, and make-level accuracy.
Uncertainty is reported two ways: bootstrap confidence intervals over the test set, and
standard deviation across five training seeds.

Model comparisons use a **paired** bootstrap, because both models see the same images and
make correlated errors. Where intervals overlap, the difference is reported as
unresolved rather than as a ranking.

## Limitations

**Error structure must be measured.** The generated error breakdown in `RESULTS.md`
will distinguish confusions between closely related classes from cross-make mistakes. In many of these cases the
distinction is a badge or a bumper detail that is not resolvable at 224×224, and
sometimes not resolvable from the photograph at all.

**No augmentation was used.** Caching embeddings from a frozen backbone precludes
augmenting the training images, since augmentation would change the embedding each epoch.
This costs some accuracy and probably some robustness.

**The model is a closed set.** Shown a car outside the 196 classes — or a photograph with
no car in it — it will still return a confident-looking distribution over car classes.
There is no abstain option and no out-of-distribution detection. Calibration improves the
honesty of probabilities *within* the distribution; it does nothing about inputs outside
it.

**Inherited backbone biases.** The representation comes from a large pretrained model
(DINOv2 or CLIP) trained on web-scale image data, and carries whatever biases and gaps
that data has. None of that was audited here.

## Calibration

A single temperature is fit on validation by minimising NLL and applied to test. This
cannot change accuracy — dividing logits by a positive scalar leaves the argmax
unchanged — but its effect on expected calibration error must be measured on the held-out test set. The demo reports
calibrated probabilities for that reason: an uncalibrated 196-class softmax reads "99%"
on cars the model cannot reliably identify, and a confidence number that cannot be
trusted is worse than showing none.

## Reproducibility

Seeded at every entry point; splits committed as CSVs; each run persists its checkpoint,
resolved config, metrics and git SHA. The embedding cache is content-addressed on the
backbone, weights tag, preprocessing spec and image ids, so cached features cannot drift
out of sync with the configuration that claims to have produced them.

## License

Code: MIT. The Stanford Cars dataset is subject to its own terms and is not redistributed
here.
