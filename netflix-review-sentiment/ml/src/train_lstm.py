import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from data_loader import (
    train_loader,
    test_loader,
    vocab,
    MAX_LENGTH
)

from lstm_model import SentimentLSTM


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

VOCAB_SIZE = len(vocab.word_to_index)

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

EPOCHS = 5
LEARNING_RATE = 0.001


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

model = model.to(device)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, lengths, y in train_loader:

        X = X.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(
            X,
            lengths
        )

        logits = logits.squeeze(1)

        # Calculate loss
        loss = criterion(
            logits,
            y
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Predictions
        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

    train_accuracy = correct / total

    average_loss = (
        total_loss / len(train_loader)
    )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for X, lengths, y in test_loader:

            X = X.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(
                X,
                lengths
            )

            logits = logits.squeeze(1)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            test_correct += (
                predictions == y
            ).sum().item()

            test_total += y.size(0)

    test_accuracy = (
        test_correct / test_total
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        model_path = (
            MODEL_DIR / "best_lstm.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word_to_index,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "max_length": MAX_LENGTH,
                "test_accuracy": test_accuracy
            },
            model_path
        )

        print(
            f"✓ Best model saved to: {model_path}"
        )


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\nTraining complete!")

print(
    f"Best Test Accuracy: "
    f"{best_accuracy:.4f}"
)