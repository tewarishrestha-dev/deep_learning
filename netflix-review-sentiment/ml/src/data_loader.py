import pandas as pd
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from tokenizer import tokenize
from vocabulary import Vocabulary


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "IMDB_Dataset.csv"


# -----------------------------
# Configuration
# -----------------------------

MAX_LENGTH = 200
BATCH_SIZE = 32


# -----------------------------
# Dataset
# -----------------------------

class ReviewDataset(Dataset):

    def __init__(self, reviews, labels, vocab, max_length):

        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):

        review = self.reviews[index]
        label = self.labels[index]

        # Tokenize
        tokens = tokenize(review)

        # Convert tokens → IDs
        token_ids = [
            self.vocab.word_to_index.get(
                token,
                self.vocab.word_to_index["<UNK>"]
            )
            for token in tokens
        ]

        # Keep original length before padding
        length = min(len(token_ids), self.max_length)

        # Truncate
        token_ids = token_ids[:self.max_length]

        # Pad
        token_ids += [
            self.vocab.word_to_index["<PAD>"]
        ] * (self.max_length - len(token_ids))

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nClass distribution:")
print(df["sentiment"].value_counts())


# -----------------------------
# Convert labels
# -----------------------------

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


# -----------------------------
# Train / Test Split
# -----------------------------

train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    df["review"].values,
    df["sentiment"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"].values
)


print("\nTraining samples:", len(train_reviews))
print("Testing samples:", len(test_reviews))


# -----------------------------
# Build vocabulary
# ONLY training data
# -----------------------------

vocab = Vocabulary()

vocab.build(train_reviews)

print("\nVocabulary size:")
print(len(vocab.word_to_index))


# -----------------------------
# Create datasets
# -----------------------------

train_dataset = ReviewDataset(
    train_reviews,
    train_labels,
    vocab,
    MAX_LENGTH
)

test_dataset = ReviewDataset(
    test_reviews,
    test_labels,
    vocab,
    MAX_LENGTH
)


# -----------------------------
# Create DataLoaders
# -----------------------------

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


# -----------------------------
# Test one batch
# -----------------------------

X_batch, lengths_batch, y_batch = next(iter(train_loader))

print("\nX batch shape:")
print(X_batch.shape)

print("\nLengths batch shape:")
print(lengths_batch.shape)

print("\nY batch shape:")
print(y_batch.shape)

print("\nFirst review token IDs:")
print(X_batch[0])

print("\nFirst review actual length:")
print(lengths_batch[0])

print("\nFirst label:")
print(y_batch[0])