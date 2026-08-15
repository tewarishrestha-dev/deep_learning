import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
DATA_DIR = "data"


transform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


if __name__ == "__main__":

    print("=" * 60)
    print("FASHION-MNIST DATASET")
    print("=" * 60)

    print("Training samples:", len(train_dataset))
    print("Test samples    :", len(test_dataset))

    X, y = next(iter(train_loader))

    print("\nBatch:")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)

    print("\nPixel range:")
    print("Minimum:", X.min().item())
    print("Maximum:", X.max().item())

    print("\nDevice available:")
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))