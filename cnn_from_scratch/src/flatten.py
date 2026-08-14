import numpy as np


class Flatten:

    def forward(self, X):

        self.input_shape = X.shape

        batch_size = X.shape[0]

        return X.reshape(batch_size, -1)


    def backward(self, d_output):

        return d_output.reshape(self.input_shape)