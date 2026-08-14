import numpy as np


class SoftmaxCrossEntropy:

    def forward(self, logits, y_true):
        self.y_true = y_true

        # Numerical stability
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp_values = np.exp(shifted)
        self.probs = exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

        batch_size = logits.shape[0]

        loss = -np.sum(
            y_true * np.log(self.probs + 1e-12)
        ) / batch_size

        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]

        return (
            self.probs - self.y_true
        ) / batch_size