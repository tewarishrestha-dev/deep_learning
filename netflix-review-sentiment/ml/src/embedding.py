import torch
import torch.nn as nn

vocab_size = 18
embedding_dim = 4

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=0
)

print("Embedding matrix:")
print(embedding.weight)

print("\nEmbedding matrix shape:")
print(embedding.weight.shape)

# Our encoded sentence
sequence = torch.tensor([13, 14, 4, 15, 0, 0])

output = embedding(sequence)

# Artificial loss
loss = output.sum()

print("Loss:")
print(loss)

# Backpropagation
loss.backward()

print("\nEmbedding gradients:")
print(embedding.weight.grad)

print("\nEmbedding output:")
print(output)

# Manually retrieve the rows
print("\nManual row lookup:")
print(embedding.weight[13])
print(embedding.weight[14])
print(embedding.weight[4])
print(embedding.weight[15])