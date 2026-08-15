import torch

from model import CNN


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model
# ============================================================

model = CNN(
    num_classes=10
).to(device)


# ============================================================
# Fake batch
# ============================================================

X = torch.randn(
    64, 1, 28, 28
).to(device)


# ============================================================
# Forward
# ============================================================

output = model(X)


# ============================================================
# Results
# ============================================================

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


# ============================================================
# Parameters
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:", total_parameters)


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 60)

if output.shape == (64, 10):
    print("✓ Forward pass successful")
else:
    print("✗ Output shape incorrect")

print("=" * 60)