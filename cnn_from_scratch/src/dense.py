import numpy as np


class Dense:

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        # He initialization
        self.W = (
            np.random.randn(input_size, output_size)
            * np.sqrt(2.0 / input_size)
        )

        self.b = np.zeros((1, output_size))


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, X):

        self.X = X

        return X @ self.W + self.b


    # ========================================================
    # Backward
    # ========================================================

    def backward(self, d_output):

        # Gradient with respect to weights
        self.dW = self.X.T @ d_output

        # Gradient with respect to bias
        self.db = np.sum(
            d_output,
            axis=0,
            keepdims=True
        )

        # Gradient with respect to input
        dX = d_output @ self.W.T

        return dX