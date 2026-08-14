import numpy as np
class ReLU:
    def forward(self, X):

        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dX):
        return dX * (self.X > 0)