import torch
import torch.nn as nn
from pathlib import Path

from tokenizer import tokenize
from lstm_model import SentimentLSTM


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_lstm.pth"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# --------------------------------------------------
# Load vocabulary
# --------------------------------------------------

word_to_index = checkpoint["vocab"]

VOCAB_SIZE = len(word_to_index)

EMBEDDING_DIM = checkpoint["embedding_dim"]
HIDDEN_DIM = checkpoint["hidden_dim"]
MAX_LENGTH = checkpoint["max_length"]


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_sentiment(review):

    tokens = tokenize(review)

    token_ids = [
        word_to_index.get(
            token,
            word_to_index["<UNK>"]
        )
        for token in tokens
    ]

    # Actual length before padding/truncation
    length = min(
        len(token_ids),
        MAX_LENGTH
    )

    # Truncate
    token_ids = token_ids[:MAX_LENGTH]

    # Pad
    token_ids += [
        word_to_index["<PAD>"]
    ] * (
        MAX_LENGTH - len(token_ids)
    )

    # Convert to tensors
    X = torch.tensor(
        [token_ids],
        dtype=torch.long
    ).to(device)

    lengths = torch.tensor(
        [length],
        dtype=torch.long
    ).to(device)

    # Model prediction
    with torch.no_grad():

        logits = model(
            X,
            lengths
        )

        probability = torch.sigmoid(
            logits
        ).item()

    # Sentiment
    if probability >= 0.5:

        sentiment = "positive"
        confidence = probability

    else:

        sentiment = "negative"
        confidence = 1 - probability

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_probability": probability,
        "negative_probability": 1 - probability
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    reviews = [
        "This movie was absolutely fantastic. I loved every minute of it.",

        "This was one of the worst movies I have ever watched.",

        "The acting was brilliant and the story was incredibly engaging."
    ]

    for review in reviews:

        result = predict_sentiment(review)

        print("\nReview:")
        print(review)

        print("\nPrediction:")
        print(result)