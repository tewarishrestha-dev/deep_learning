import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
)

os.makedirs("images", exist_ok=True)

def plot_loss(loss_history):

    plt.figure(figsize=(7,5))
    plt.plot(loss_history, linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "loss_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred):

    plt.figure(figsize=(6,6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))
    plt.close()


def plot_roc(y_true, y_prob):

    plt.figure(figsize=(6,6))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "roc_curve.png"))
    plt.close()


def plot_precision_recall(y_true, y_prob):

    plt.figure(figsize=(6,6))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob
    )

    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "precision_recall_curve.png"))
    plt.close()


def plot_probability_distribution(y_prob):

    plt.figure(figsize=(7,5))
    plt.hist(y_prob, bins=25)
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Probability of Churn")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "probability_distribution.png"))
    plt.close()


def plot_weight_distribution(model):

    weights = np.concatenate([
        model.W1.ravel(),
        model.W2.ravel()
    ])

    plt.figure(figsize=(7,5))
    plt.hist(weights, bins=30)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "weight_distribution.png"))
    plt.close()

def plot_accuracy(train_acc,test_acc):

    plt.figure(figsize=(5,4))
    plt.bar(["Train","Test"],[train_acc,test_acc])
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR,"accuracy.png"))
    plt.close()    

def plot_optimizer_comparison(gd_loss,adam_loss):

    plt.figure(figsize=(7,5))
    plt.plot(gd_loss,label="Gradient Descent",linewidth=2)
    plt.plot(adam_loss,label="Adam",linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR,"optimizer_comparison.png"))
    plt.close()    