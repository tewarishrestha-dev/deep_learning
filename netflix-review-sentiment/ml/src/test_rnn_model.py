import torch
from rnn_model import SentimentRNN


# Model parameters
vocab_size = 18
embedding_dim = 4
hidden_dim = 8

model = SentimentRNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# Simulate a batch of 3 reviews
# Each review contains 6 tokens
x = torch.tensor([
    [13, 14, 4, 15, 0, 0],
    [2, 3, 4, 6, 0, 0],
    [13, 16, 4, 17, 0, 0]
])

print("Input:")
print(x)

print("\nInput shape:")
print(x.shape)

output = model(x)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)