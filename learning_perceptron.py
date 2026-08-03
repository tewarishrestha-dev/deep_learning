import numpy as np

X = np.array([
    [2, 30],
    [3, 40],
    [5, 70],
    [6, 90]
])

y = np.array([0, 0, 1, 1])

w = np.random.randn(2)
b = np.random.randn()

lr = 0.1

epochs = 20

for epoch in range(epochs):

    for x_i, y_i in zip(X, y):

        z = np.dot(w, x_i) + b

        prediction = 1 if z >= 0 else 0

        w += lr * (y_i - prediction) * x_i

        b += lr * (y_i - prediction)

print("Weights:", w)
print("Bias:", b)