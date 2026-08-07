import argparse
import numpy as np
from preprocess import load_data
from model import NeuralNetwork
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from plot import (
    plot_loss,
    plot_confusion_matrix,
    plot_roc,
    plot_precision_recall,
    plot_probability_distribution,
    plot_weight_distribution
)

# ARGUMENTS

parser = argparse.ArgumentParser(
    description="Train Neural Network from Scratch"
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="gd",
    choices=["gd", "adam"],
    help="Optimizer to use (gd or adam)"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.35,
    help="Classification threshold"
)

args = parser.parse_args()

# LOAD DATA

X_train, X_test, y_train, y_test = load_data()

# CREATE MODEL

model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=64,
    output_size=1
)

# CLASS WEIGHTS

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train.ravel()
)
model.class_weight = weights
print(f"\nUsing Optimizer : {args.optimizer.upper()}")
print(f"Class weights   : {weights}")

# OPTIMIZER SETTINGS

if args.optimizer == "adam":
    lr = 0.001
else:
    lr = 0.05

epochs = 2000
threshold = args.threshold

print(f"Learning Rate   : {lr}")
print(f"Epochs          : {epochs}")
print(f"Threshold       : {threshold}")

# TRAIN

model.fit(
    X_train,
    y_train,
    epochs=epochs,
    lr=lr,
    optimizer=args.optimizer
)

# PREDICTIONS

y_prob = model.predict_proba(X_test).ravel()
y_pred = model.predict(
    X_test,
    threshold = threshold
).ravel()

# ACCURACY

train_acc = model.accuracy(X_train, y_train)
test_acc = np.mean(
    y_pred == y_test.ravel()
)

# RESULTS

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")

# CLASSIFICATION REPORT

print("\nClassification Report\n")

print(
    classification_report(
        y_test.ravel(),
        y_pred,
        zero_division=0
    )
)

# CONFUSION MATRIX

print("Confusion Matrix\n")

print(
    confusion_matrix(
        y_test.ravel(),
        y_pred
    )
)

# PROBABILITY STATISTICS

print("\nProbability Statistics")
print("Min:", y_prob.min())
print("Max:", y_prob.max())
print("Mean:", y_prob.mean())

# PREDICTION DISTRIBUTION

print("\nPrediction Distribution")
print(
    np.unique(
        y_pred,
        return_counts=True
    )
)

# PLOTS

plot_loss(model.loss_history)
plot_confusion_matrix(
    y_test.ravel(),
    y_pred
)
plot_roc(
    y_test.ravel(),
    y_prob
)
plot_precision_recall(
    y_test.ravel(),
    y_prob
)
plot_probability_distribution(
    y_prob
)
plot_weight_distribution(
    model
)