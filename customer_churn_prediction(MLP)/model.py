import numpy as np

class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)

        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)

        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        self.loss_history = []
        self.class_weight = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, y):
        y = y.reshape(-1,1)

        eps = 1e-8
        if self.class_weight is None:
            loss = -np.mean(
                y*np.log(self.A2 + eps)
                +
                (1-y)*np.log(1-self.A2 + eps)
            )

        else:
            w0, w1 = self.class_weight
            loss = -np.mean(
                w1*y*np.log(self.A2 + eps)
                +
                w0*(1-y)*np.log(1-self.A2 + eps)
            )
        return loss

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        if self.class_weight is None:
            self.dZ2 = self.A2 - y
        else:
            w0, w1 = self.class_weight
            sample_weights = np.where(
                y == 1,
                w1,
                w0
            )
            self.dZ2 = (
                self.A2 - y
            ) * sample_weights

        self.dW2 = (
            self.A1.T @ self.dZ2
        ) / m

        self.db2 = np.sum(
            self.dZ2,
            axis=0,
            keepdims=True
        ) / m

        self.dZ1 = (
            self.dZ2 @ self.W2.T
        ) * (self.Z1 > 0)

        self.dW1 = (
            X.T @ self.dZ1
        ) / m

        self.db1 = np.sum(
            self.dZ1,
            axis=0,
            keepdims=True
        ) / m

    def update_parameters(self, lr):

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1

        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def update_parameters_adam(self, lr):

        self.t += 1
        self.mW1 = (
            self.beta1*self.mW1
            +
            (1-self.beta1)*self.dW1
        )
        self.mb1 = (
            self.beta1*self.mb1
            +
            (1-self.beta1)*self.db1
        )
        self.mW2 = (
            self.beta1*self.mW2
            +
            (1-self.beta1)*self.dW2
        )
        self.mb2 = (
            self.beta1*self.mb2
            +
            (1-self.beta1)*self.db2
        )

        self.vW1 = (
            self.beta2*self.vW1
            +
            (1-self.beta2)*(self.dW1**2)
        )
        self.vb1 = (
            self.beta2*self.vb1
            +
            (1-self.beta2)*(self.db1**2)
        )
        self.vW2 = (
            self.beta2*self.vW2
            +
            (1-self.beta2)*(self.dW2**2)
        )
        self.vb2 = (
            self.beta2*self.vb2
            +
            (1-self.beta2)*(self.db2**2)
        )

        mW1_hat = self.mW1 / (1-self.beta1**self.t)
        mb1_hat = self.mb1 / (1-self.beta1**self.t)

        mW2_hat = self.mW2 / (1-self.beta1**self.t)
        mb2_hat = self.mb2 / (1-self.beta1**self.t)


        vW1_hat = self.vW1 / (1-self.beta2**self.t)
        vb1_hat = self.vb1 / (1-self.beta2**self.t)

        vW2_hat = self.vW2 / (1-self.beta2**self.t)
        vb2_hat = self.vb2 / (1-self.beta2**self.t)

        self.W1 -= lr * mW1_hat / (
            np.sqrt(vW1_hat) + self.epsilon
        )
        self.b1 -= lr * mb1_hat / (
            np.sqrt(vb1_hat) + self.epsilon
        )
        self.W2 -= lr * mW2_hat / (
            np.sqrt(vW2_hat) + self.epsilon
        )
        self.b2 -= lr * mb2_hat / (
            np.sqrt(vb2_hat) + self.epsilon
        )

    def fit(
        self,
        X,
        y,
        epochs=1000,
        lr=0.01,
        optimizer="gd"
    ):
        
        self.loss_history = []

        for epoch in range(epochs):

            self.forward(X)
            loss = self.compute_loss(y)
            self.loss_history.append(loss)

            self.backward(X,y)
            if optimizer=="adam":
                self.update_parameters_adam(lr)
            else:
                self.update_parameters(lr)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:4d} | Loss: {loss:.4f}"
                )


    def predict_proba(self,X):
        return self.forward(X)

    def predict(self,X,threshold=0.5):
        return (
            self.predict_proba(X)
            >= threshold
        ).astype(int)

    def accuracy(self,X,y):
        predictions = self.predict(X)
        return np.mean(
            predictions == y.reshape(-1,1)
        )