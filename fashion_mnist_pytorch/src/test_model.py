import torch
from fashion_mnist_pytorch.experiments.model import CNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN(
    num_classes=10
).to(device)

X = torch.randn(
    64, 1, 28, 28
).to(device)

output = model(X)

print("=" * 60)
print("PYTORCH CNN TEST")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nInput shape :", X.shape)
print("Output shape:", output.shape)

print("\nExpected output:")
print("(64, 10)")

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:", total_parameters)

print("\n" + "=" * 60)

if output.shape == (64, 10):
    print("✓ Forward pass successful")
else:
    print("✗ Output shape incorrect")

print("=" * 60)