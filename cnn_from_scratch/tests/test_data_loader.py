from data_loader import load_fashion_mnist

X_train, y_train, X_test, y_test = load_fashion_mnist(
    "data/fashion-mnist_train.csv",
    "data/fashion-mnist_test.csv"
)

print("=" * 50)
print("FASHION-MNIST DATASET")
print("=" * 50)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("Min pixel:", X_train.min())
print("Max pixel:", X_train.max())

print("First label:", y_train[0])
print("First image:", X_train[0].shape)