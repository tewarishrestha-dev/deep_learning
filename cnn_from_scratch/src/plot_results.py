import numpy as np
import matplotlib.pyplot as plt

loss = np.load("loss_history.npy")
train_acc = np.load("train_accuracy_history.npy")
test_acc = np.load("test_accuracy_history.npy")

epochs = np.arange(1, len(loss) + 1)

# Loss
plt.figure()
plt.plot(epochs, loss, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.xticks(epochs)
plt.grid(True)
plt.savefig("loss.png", dpi=150, bbox_inches="tight")
plt.show()

# Accuracy
plt.figure()
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, test_acc, marker="o", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.savefig("accuracy.png", dpi=150, bbox_inches="tight")
plt.show()