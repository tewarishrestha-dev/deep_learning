# Customer Churn Prediction using MLP from Scratch

A customer churn prediction model implemented **from scratch using NumPy** without using deep learning frameworks like TensorFlow or PyTorch.

This project implements a **Multi-Layer Perceptron (MLP)** for binary classification and compares the performance of:

* Gradient Descent (GD)
* Adam Optimizer

The goal is to understand neural networks, backpropagation, and optimization algorithms by building them manually.

---

## Features

* Neural Network implemented from scratch using NumPy
* Forward propagation
* Backpropagation
* ReLU activation function
* Sigmoid output layer
* He weight initialization
* Binary Cross-Entropy loss
* Class weighting for imbalanced data
* Gradient Descent optimizer
* Adam optimizer
* Model evaluation and visualization

---

## Model Architecture

```
Input Features
      |
      v
Dense Layer (64 neurons)
      |
     ReLU
      |
      v
Output Layer (1 neuron)
      |
   Sigmoid
      |
      v
Churn Probability
```

---

## Dataset

Dataset used:

**Telco Customer Churn Dataset**

The model predicts:

```
0 → Customer stays
1 → Customer churns
```

Preprocessing includes:

* Removing unnecessary columns
* Handling missing values
* Encoding categorical features
* Feature scaling

---

## Optimizers Compared

### Gradient Descent

Standard parameter update:

[
\theta = \theta - \eta \nabla L
]

### Adam Optimizer

Implemented using:

* Momentum
* Adaptive learning rates
* Bias correction

---

## Results

### Gradient Descent

```
Test Accuracy : 73.42%
```

### Adam Optimizer

```
Test Accuracy : 74.41%
```

Adam showed better convergence and improved performance compared to vanilla Gradient Descent.

---

## Visualizations

The project generates:

* Loss curve
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Probability distribution
* Weight distribution

---

## Project Structure

```
customer_churn_prediction(MLP)
│
├── model.py
├── train.py
├── preprocess.py
├── plot.py
├── README.md
├── requirements.txt
└── images/
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with Gradient Descent:

```bash
python train.py --optimizer gd --threshold 0.50 
```

Run with Adam:

```bash
python train.py --optimizer adam --threshold 0.50
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Learning Outcomes

Through this project, I implemented and learned:

* Neural networks from first principles
* Backpropagation
* Gradient-based optimization
* Adam optimizer
* Handling class imbalance
* Model evaluation techniques

---

## Future Improvements

* Mini-batch Gradient Descent
* Dropout regularization
* Early stopping
* Hyperparameter tuning
* Comparison with PyTorch implementation

```
```

