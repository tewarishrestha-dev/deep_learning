import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits