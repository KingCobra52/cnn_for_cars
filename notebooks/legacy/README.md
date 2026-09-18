# Legacy notebook — the "before" state

`cifar_cnn_vs_vgg16.ipynb` is the original project, preserved deliberately. It was
committed as `final_obj_detection_notebook-4.ipynb`: a single 304 KB Colab notebook that
trained a small CNN and a frozen VGG16 head on a 3-class subset of CIFAR-10
(`background` / `car` / `truck`, 32x32) and reported 82.8% and 85.5%.

It is kept here, with its outputs stripped, because the bugs in it are the reason the
rest of this repository is shaped the way it is. Each one below is a real defect found in
the committed state, followed by what the rebuild does instead.

## 1. The committed notebook did not run

Cell 32 plots a variable named `vgg16_losses`. The training loop in cell 24 accumulates
into `vgg_16_losses` — different name, with an underscore. `vgg16_losses` is defined
nowhere in the notebook. On a clean kernel this is a `NameError`; the figure only exists
in the committed outputs because a previous run had left the name bound in kernel state.

Cell 30 ships an unresolved traceback:

```
ValueError: x and y must have same first dimension, but have shapes (5,) and (1,)
```

It plots five epochs against a `train_losses` list holding one element, because
`train_losses.append(...)` sits outside the epoch loop.

**Now:** every notebook is stripped of outputs by a pre-commit hook and verified in CI, so
a figure can never again be "real" only because of stale kernel state. All logic lives in
`src/carvision/`, is imported by the notebooks, and is covered by tests.

## 2. Every loss curve was meaningless

Both training loops append to the loss list inside the **batch** loop rather than the
epoch loop, and the CNN's `running_loss` is initialised once before the epoch loop and
never reset. So `cnn_losses` holds 938 monotonically increasing values — a running total
divided by a constant — not two epoch losses. The "Training Loss Comparison" figure plots
that.

**Now:** metrics are accumulated by explicit per-epoch reduction in
`carvision.train`, written to `metrics.json`, and logged to MLflow. A test asserts the
recorded epoch count matches the configured one.

## 3. Reported numbers disagreed with plotted numbers

Cell 18 prints `accuracy 0.82833331823349`. Cells 29 and 30 plot the CNN at a hardcoded
`83.26` — a stale number from some earlier run, typed into a list literal:

```python
accuracies = [85.53, 83.26]
```

**Now:** no number is ever typed by hand. `docs/RESULTS.md` is generated from the MLflow
run table, figures are generated from the same JSON the metrics come from, and every
headline accuracy carries a bootstrap confidence interval.

## 4. A pretrained ImageNet model was fed the wrong input

VGG16 was loaded pretrained, then given 32x32 tensors normalised with CIFAR-10
statistics. The model expects 224x224 input normalised with ImageNet statistics. It still
reached 85.5% because the task is nearly trivial, which is exactly what makes the mistake
easy to miss.

**Now:** preprocessing is owned by the backbone. `carvision.data.transforms` returns the
transform that the backbone's own weights enum declares, so input size and normalisation
statistics cannot drift apart from the weights.

## 5. Nothing was reproducible and nothing was saved

No `random`, `numpy` or `torch` seed anywhere, with `shuffle=True` on the train loader —
so no two runs produced the same number. `torch.save` appears zero times, so both trained
models were discarded when the kernel died. No dependency file, so the environment was
whatever Colab happened to ship that day.

**Now:** `carvision.utils.seed.set_seed` is called at every entry point, every run writes
a checkpoint plus its fully resolved config and the git SHA, and dependencies are pinned
in `requirements.lock`.

## 6. Smaller things, still worth naming

- Trained on CPU inside a GPU runtime. The loop carries the comment
  `# No need to move to device, Colab handles this`, which is false — nothing calls
  `.to(device)`. Each VGG16 epoch took ~16 minutes.
- `models.vgg16(pretrained=True)`: deprecated since torchvision 0.13 and warning on every
  run.
- The dataset was loaded and re-preprocessed three times with copy-pasted blocks; the
  20-line confusion-matrix unpacking block appears twice verbatim, rebuilding an array
  `confusion_matrix()` had already returned.
- The `background` class is the first N images of the other eight CIFAR classes, taken
  unshuffled — so it is biased toward CIFAR's storage order rather than being a balanced
  sample. The rebuild's `cifar_smoke` dataset fixes this with a seeded stratified sample.
- No validation split at all. The test set was the only held-out data, and it was used for
  the headline numbers.
- Dead code throughout: `plot_acc` is Keras-specific (`history.history['val_accuracy']`)
  in a PyTorch notebook and could never have run; it carries the comment
  `# i'm sorry for this function's code. i am so sorry.`
- The whole file was named for object detection. It has never contained any.
