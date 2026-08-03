import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

def plot_decision_boundary(model, X, y, epoch):
    print("Visualization function called!")

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.predict(point) for point in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[y==0,0], X[y==0,1], label="Setosa")
    plt.scatter(X[y==1,0], X[y==1,1], label="Not Setosa")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title(f"Epoch {epoch}")
    plt.legend()
    plt.savefig(os.path.join(IMAGE_DIR, f"epoch_{epoch:02d}.png"))
    plt.close()

def plot_accuracy(history):

    plt.figure(figsize=(7,5))

    plt.plot(range(1, len(history)+1), history, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.grid(True)

    plt.savefig(os.path.join(IMAGE_DIR, "accuracy.png"))

    plt.close()