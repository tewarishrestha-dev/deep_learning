import numpy as np

def predict(x, w, b):
    z = np.dot(w, x) + b
    return 1 if z >= 0 else 0

x = np.array([3, 20])

w = np.array([1.5, -0.1])

b = -1
print(z)
print(predict(x, w, b))