# Customer Churn Prediction using a Neural Network from Scratch

A binary classification neural network implemented **from scratch using NumPy** to predict whether a telecom customer will churn.

---

## Dataset

- **Dataset:** Telco Customer Churn
- **Samples:** 7043
- **Features:** 40 (after preprocessing)
- **Target:** Churn (0 = No, 1 = Yes)

---

## Project Structure

```
customer_churn_prediction/
│
├── data/
├── model.py
├── preprocess.py
├── train.py
├── plot.py
├── images/
└── README.md
```

---

## Neural Network Architecture

```
Input Layer (40 Features)
          │
          ▼
Hidden Layer (16 Neurons)
     Sigmoid Activation
          │
          ▼
Output Layer (1 Neuron)
     Sigmoid Activation
          │
          ▼
 Churn Probability
```

---

# Forward Propagation

### Hidden Layer

\[
Z_1 = XW_1 + b_1
\]

\[
A_1 = \sigma(Z_1)
\]

---

### Output Layer

\[
Z_2 = A_1W_2 + b_2
\]

\[
A_2 = \sigma(Z_2)
\]

---

### Sigmoid Function

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

---

# Binary Cross Entropy Loss

\[
L=-\frac1m\sum
\left[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\right]
\]

---

# Backpropagation

### Output Layer

\[
dZ_2=A_2-y
\]

\[
dW_2=\frac1mA_1^TdZ_2
\]

\[
db_2=\frac1m\sum dZ_2
\]

---

### Hidden Layer

\[
dZ_1=(dZ_2W_2^T)\times A_1(1-A_1)
\]

\[
dW_1=\frac1mX^TdZ_1
\]

\[
db_1=\frac1m\sum dZ_1
\]

---

# Gradient Descent

\[
W=W-\alpha dW
\]

\[
b=b-\alpha db
\]

where

- **α** = Learning Rate

---

# Training Pipeline

```
Load Dataset
      ↓
Preprocessing
      ↓
Forward Propagation
      ↓
Binary Cross Entropy
      ↓
Backpropagation
      ↓
Gradient Descent
      ↓
Repeat for N Epochs
      ↓
Prediction
      ↓
Evaluation
```

---

# Results

- Train Accuracy: **79.9%**
- Test Accuracy: **79.5%**

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Future Improvements

- ReLU Activation
- Xavier / He Initialization
- Adam Optimizer
- Mini-batch Gradient Descent
- Dropout
- TensorFlow / Keras Implementation

---

