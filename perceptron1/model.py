import numpy as np

class Perceptron:

    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def step(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return self.step(z)

    def fit(self, X, y, plot_callback=None):

        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.history = []
        frame = 1

        for epoch in range(self.epochs):

            for x_i, y_i in zip(X, y):

                prediction = self.predict(x_i)
                update = self.lr * (y_i - prediction)
                self.w += update * x_i
                self.b += update

            if plot_callback:
                plot_callback(
                    self,
                    X,
                    y,
                    frame
                )
                frame += 1

            acc = self.score(X, y)
            self.history.append(acc)
            print(
                f"Epoch {epoch+1}/{self.epochs} | Accuracy: {acc:.4f}"
            )

    def score(self, X, y):
        predictions = []
        for x in X:
            predictions.append(self.predict(x))
        predictions = np.array(predictions)
        return np.mean(predictions == y)