# CNN From Scratch

A Convolutional Neural Network implemented completely from scratch using NumPy, without TensorFlow or PyTorch.

## Features

- Conv2D forward & backward propagation
- ReLU activation
- Max Pooling
- Flatten layer
- Dense layers
- Softmax + Cross-Entropy Loss
- Gradient checking
- Mini-batch Gradient Descent
- Vectorized Conv2D
- Fashion-MNIST classification

## Architecture

28×28×1 Image
↓
Conv2D (16 filters)
↓
ReLU
↓
MaxPool
↓
Conv2D (32 filters)
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Dense (128)
↓
ReLU
↓
Dense (10)
↓
Softmax + Cross Entropy

## Results

Training on 5,000 Fashion-MNIST images for 3 epochs.

| Metric | Result |
|---|---:|
| Training Accuracy | 61.0% |
| Test Accuracy | 60.9% |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 3 |

### Training Loss

2.1815 → 1.6768 → 1.3473

## Speed Improvement

The original loop-based Conv2D took approximately 25 minutes.

The vectorized Conv2D reduced training time to approximately 5–7 minutes while passing numerical gradient checks.

## Dataset

Fashion-MNIST:

https://github.com/zalandoresearch/fashion-mnist

Place the following files inside the `data/` folder:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

## Run

```bash
pip install -r requirements.txt
python src/train.py