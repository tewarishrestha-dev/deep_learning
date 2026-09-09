import torch
import torch.nn as nn


class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super().__init__()

        # -----------------------------
        # Embedding layer
        # -----------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # -----------------------------
        # Vanilla RNN
        # -----------------------------

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # -----------------------------
        # Classification layer
        # -----------------------------

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x, lengths):

        # x:
        # [batch_size, sequence_length]

        embedded = self.embedding(x)

        # embedded:
        # [batch_size, sequence_length, embedding_dim]

        output, hidden = self.rnn(embedded)

        # output:
        # [batch_size, sequence_length, hidden_dim]

        # ----------------------------------
        # Find the final REAL token
        # ----------------------------------

        last_indices = lengths - 1

        batch_indices = torch.arange(
            x.size(0),
            device=x.device
        )

        final_hidden = output[
            batch_indices,
            last_indices,
            :
        ]

        # final_hidden:
        # [batch_size, hidden_dim]

        logits = self.fc(final_hidden)

        # logits:
        # [batch_size, 1]

        return logits