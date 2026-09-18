# Before and after

This repository began as one Colab notebook. Each defect in it maps to something in the
current codebase that makes the defect structurally impossible rather than merely
unlikely. That mapping is most of why the architecture looks the way it does.

The notebook itself is preserved at
[`notebooks/legacy/cifar_cnn_vs_vgg16.ipynb`](../notebooks/legacy/README.md).

| Then | Now |
| --- | --- |
| A figure plotted `vgg16_losses`, defined nowhere. It rendered only because a previous run had left the name bound in kernel state. | All logic lives in `src/carvision/` and is imported, tested, and covered by CI. A pre-commit hook strips notebook outputs and CI fails if any notebook carries them, so no figure can be "real" only because of stale state. |
| Loss lists appended inside the batch loop, with `running_loss` never reset — so the "loss curve" was a running total over 938 batches. | `train.py` accumulates a weighted sum and divides once at the end of the epoch. Exactly one `EpochRecord` is appended per epoch, at one place in the code. |
| Plots hardcoded `83.26` while the model had scored `82.83`. | No number is typed by hand. `figures.py` reads `evaluation.json` and the saved predictions, so a plot cannot disagree with the metrics it illustrates. |
| A pretrained ImageNet VGG16 was fed 32×32 CIFAR-normalised tensors. It still scored 85%, which is what made the bug invisible. | `BackboneSpec` binds a `PreprocessSpec` to the weights it belongs to. No transform is constructed by hand, and the spec is hashed into the cache key. |
| No seeds anywhere, with `shuffle=True`. No two runs agreed. | `set_seed` at every entry point, covering `random`, `numpy` and `torch`, with deterministic kernels. A test asserts identical logits from identical seeds. |
| `torch.save` appears zero times. Both models were lost when the kernel died. | Every run writes a checkpoint, its fully resolved config, metrics, and the git SHA. |
| No validation split. The test set was used for tuning *and* for the headline number. | The official test split is untouched. A seeded stratified 15% validation set is carved from train, committed as CSVs, and `eval.py` is the only module that reads test. |
| No dependency file — the environment was whatever Colab shipped that day. | `pyproject.toml` with pinned bounds, `requirements.lock`, and CI on Python 3.11 and 3.12. |
| Trained on CPU inside a GPU runtime, under a comment claiming `Colab handles this`. ~16 min per epoch. | Device handling is explicit, and the whole architecture is designed for the hardware that actually exists. |
| The dataset was reloaded and re-preprocessed three times with copy-pasted blocks. | One data module. One cache. `load(backbone, split)` is the entire interface. |
| `background` was the first N images of the other eight CIFAR classes, unshuffled — biased toward storage order. | The rebuilt `cifar_smoke` draws an equal share from each of the eight, with a seeded RNG, and a test asserts the balance. |
| `classification_report` imported and never called. | Called, and returned as structured data so the analysis notebook can sort and join on it. |
| The file was named for object detection. It has never contained any. | Named for what it is. |

## The three things that actually mattered

Most of the table is hygiene. Three items are different in kind, and they are the ones
worth understanding.

**1. The bug that hid because the task was too easy.** Feeding CIFAR-normalised 32×32
input to an ImageNet VGG16 is badly wrong, and the model scored 85% anyway — because
telling cars from birds does not require the features to be right. An easy benchmark does
not validate a pipeline; it conceals it. This is the strongest argument for moving to a
196-class fine-grained task, quite apart from the task being more impressive: on
Stanford Cars, a preprocessing mistake of that kind is immediately visible in the
accuracy.

**2. Tuning on the test set.** No validation split meant every choice — when to stop,
which model to report — was made using the data the result was then reported on. No
amount of care elsewhere repairs that. It is the difference between a number and a
claim.

**3. State that looked like a result.** The notebook's most alarming property was not
that it crashed. It was that it produced a plausible-looking figure from a variable that
did not exist, because an earlier run had left it in memory. Anything whose correctness
depends on execution order, in an environment that does not enforce execution order, is
not a result — and the only reliable fix is to move the logic somewhere that can be
imported and tested.

## What carried over

The original work was not worthless, and the rebuild reuses it. The 3-class CIFAR task
survives as `cifar_smoke`, a fast end-to-end test of the whole pipeline that needs no
16k-image download. The CNN-from-scratch versus transfer-learning comparison was a sound
instinct; it is now a comparison across three pretrained backbones plus a training-free
zero-shot baseline, with confidence intervals attached.
