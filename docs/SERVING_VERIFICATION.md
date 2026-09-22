# Selected-model serving verification

Verified locally on 2026-09-22 against the existing published serving bundle. The test
resolved the winner with the validation-based selection helper and selected
`dinov2_vits14-mlp-seed3`. It validated the split, training run, evaluation, serving
provenance, graph checksum, class order, and calibration temperature before inference.
The bundle was not regenerated or replaced.

- Images (six portrait and six landscape, deterministically selected): `test_00000`,
  `test_00001`, `test_00002`, `test_00003`, `test_00004`, `test_00005`, `test_00024`,
  `test_00624`, `test_00704`, `test_01573`, `test_01889`, `test_01899`
- Checkpoint SHA-256: `2afb2f56b8b11d62f574a4425bbae01b8ae971dfd4ccda31671004d2c9f35e4b`
- Evaluation SHA-256: `885ba1811bb0db125d645acd63b3631ed9ea97dd704fff398286dc1367632494`
- ONNX SHA-256: `2ee1742b32b751f0a17fa655d1273568447b21b1892a50fce461f0fde256e587`
- Calibration temperature: `0.7464581727981567`
- Preprocessing tolerance: `rtol=1e-4`, `atol=1e-4`; maximum absolute difference:
  `7.152557373046875e-07`
- Calibrated-logit tolerance: `rtol=1e-3`, `atol=1e-4`; maximum absolute difference:
  `5.817413330078125e-05`
- Runtime: Python 3.12.4, PyTorch 2.14.0, torchvision 0.29.0, ONNX 1.23.0,
  ONNX Runtime 1.30.0, NumPy 2.4.6, Pillow 11.3.0
- Outcome: **PASS**. PyTorch and ONNX top-1 predictions matched for all 12 images;
  class-name mapping matched the evaluation and dataset; probabilities were finite,
  nonnegative, and summed to one within `1e-6`.

Command:

```console
.venv/bin/pytest -s tests/test_real_model_pilot.py::test_selected_model_real_image_serving
```
