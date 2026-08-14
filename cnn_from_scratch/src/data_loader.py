import numpy as np
import pandas as pd


def load_fashion_mnist(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train.iloc[:, 0].values
    X_train = train.iloc[:, 1:].values

    y_test = test.iloc[:, 0].values
    X_test = test.iloc[:, 1:].values

    # Normalize
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Reshape: (N, 784) -> (N, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test