from sklearn.datasets import load_iris
from model import Perceptron
from visualize import plot_decision_boundary, plot_accuracy
iris = load_iris()

X = iris.data[:, 2:4]          # Petal Length, Petal Width
y = (iris.target != 0).astype(int)

model = Perceptron(lr=0.01, epochs=20)
model.fit(X, y, plot_callback=plot_decision_boundary)

print(f"Training Accuracy: {model.score(X, y):.2f}")
print("Weights:", model.w)
print("Bias:", model.b)
plot_accuracy(model.history)