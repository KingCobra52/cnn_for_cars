# README.md
# Vehicle Image Classification: CNN vs. Transfer Learning

This project compares two approaches to classifying images as `background`, `car`, or `truck`: a small convolutional neural network trained from scratch, and a VGG16 model fine-tuned via transfer learning.

## 📂 Project Structure

- `final_obj_detection_notebook-4.ipynb` — the entire project (data prep, both models, training, evaluation, visualization)

## 🚀 What it does

- Builds a 3-class dataset (`background`, `car`, `truck`) from CIFAR-10 by filtering to the car/truck classes and sampling background images from the rest
- Trains a custom CNN (2 conv layers + 2 fully-connected layers) from scratch
- Fine-tunes a pretrained VGG16 (frozen convolutional base, retrained classifier head)
- Evaluates both models with accuracy and confusion matrices
- Visualizes accuracy comparisons, training loss curves, and confusion matrix heatmaps

## 🧠 Models

| Model | Approach | Test Accuracy |
|---|---|---|
| Custom CNN | Trained from scratch (2 epochs) | ~82.8% |
| VGG16 | Transfer learning, classifier head fine-tuned (5 epochs) | ~85.5% |

## 📊 Dataset

- Source: CIFAR-10 (via `tensorflow.keras.datasets.cifar10`), downloaded automatically on first run
- Classes: `background`, `car`, `truck`
- Split: 5000 images/class (train), 1000 images/class (test)

## 🛠️ Requirements

- Python with `tensorflow`, `torch`, `torchvision`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`
- GPU recommended for faster training (especially VGG16 fine-tuning)

## ▶️ Running the Notebook

Open `final_obj_detection_notebook-4.ipynb` and run cells in order:

1. Load and construct the vehicle dataset from CIFAR-10
2. Train the custom CNN
3. Fine-tune VGG16 via transfer learning
4. Compare accuracy, training loss, and confusion matrices for both models
