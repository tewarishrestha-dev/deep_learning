import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, y):

        return -np.mean(
            y * np.log(self.A2 + 1e-8) +
            (1 - y) * np.log(1 - self.A2 + 1e-8)
        )

    def backward(self, X, y):

        m = X.shape[0]

        self.dZ2 = self.A2 - y
        self.dW2 = (self.A1.T @ self.dZ2) / m
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True) / m

        self.dZ1 = (self.dZ2 @ self.W2.T) * self.A1 * (1 - self.A1)
        self.dW1 = (X.T @ self.dZ1) / m
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True) / m

    def update_parameters(self, lr):

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1

        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def fit(self, X, y, epochs=1000, lr=0.1):

        self.loss_history = []

        for epoch in range(epochs):

            self.forward(X)

            loss = self.compute_loss(y)
            self.loss_history.append(loss)

            self.backward(X, y)
            self.update_parameters(lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)   