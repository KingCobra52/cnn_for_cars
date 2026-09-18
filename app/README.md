---
title: carvision
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# carvision

Fine-grained car recognition across the 196 classes of Stanford Cars.

A frozen self-supervised backbone with a linear probe trained on cached embeddings. The
architecture is a direct consequence of its constraint: the project was built without a
GPU, so the expensive backbone pass was made a one-time cache and the cheap head became
the experiment.

Source and full results: https://github.com/KingCobra52/cnn_for_cars
