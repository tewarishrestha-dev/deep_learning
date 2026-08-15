# CNN Image Classification — PyTorch

A Convolutional Neural Network built using **PyTorch** for image classification.

## Features

* CNN architecture using PyTorch
* Convolution, ReLU, and MaxPooling layers
* Cross-Entropy loss
* Adam optimizer
* GPU/CUDA support
* Training and validation
* Test evaluation
* Confusion matrix
* Misclassified image analysis

## Results

* **Test Accuracy:** 91.51%
* **Test Loss:** 0.2509
* **Correct Predictions:** 9151 / 10000

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

The model automatically uses CUDA when a compatible NVIDIA GPU is available.
