from preprocess import load_data
from model import NeuralNetwork
from plot import plot_loss

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from plot import (
    plot_loss,
    plot_confusion_matrix,
    plot_roc,
    plot_precision_recall,
    plot_probability_distribution,
    plot_weight_distribution
)

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Create Model
model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=1
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=1000,
    lr=0.1
)

# Accuracy
train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

y_prob = model.predict_proba(X_test).ravel()
y_pred = model.predict(X_test).ravel() 

# Plot Loss

plot_loss(model.loss_history)
plot_confusion_matrix(y_test, y_pred)
plot_roc(y_test, y_prob)
plot_precision_recall(y_test, y_prob)
plot_probability_distribution(y_prob)
plot_weight_distribution(model)