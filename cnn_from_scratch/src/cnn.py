import numpy as np
from conv2d import Conv2D
from relu import ReLU
from pooling import MaxPool2D
from flatten import Flatten
from dense import Dense


class CNN:

    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3, 1, 1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(8, 16, 3, 1, 1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        self.flatten = Flatten()

        self.fc1 = Dense(784, 128)
        self.relu3 = ReLU()
        self.fc2 = Dense(128, 10)

    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.pool1.forward(X)

        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pool2.forward(X)

        X = self.flatten.forward(X)

        X = self.fc1.forward(X)
        X = self.relu3.forward(X)
        X = self.fc2.forward(X)

        return X

    def backward(self, d_output):
        dX = self.fc2.backward(d_output)
        dX = self.relu3.backward(dX)
        dX = self.fc1.backward(dX)

        dX = self.flatten.backward(dX)

        dX = self.pool2.backward(dX)
        dX = self.relu2.backward(dX)
        dX = self.conv2.backward(dX)

        dX = self.pool1.backward(dX)
        dX = self.relu1.backward(dX)
        dX = self.conv1.backward(dX)

        return dX