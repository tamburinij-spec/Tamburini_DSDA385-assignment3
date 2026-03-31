import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM language model for text generation.
    - Pass embedding_matrix=None  → randomly initialised embeddings (one-hot proxy)
    - Pass embedding_matrix=<array> → frozen GloVe pretrained embeddings
    """
    def __init__(self, vocab_size, embed_size, hidden_size,
                 embedding_matrix=None, freeze_embeddings=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x       = self.embedding(x)       # B × T → B × T × E
        out, _  = self.lstm(x)            # B × T × H
        return self.fc(out)               # B × T × V


# ── Seq2Seq components for Task 2 ─────────────────────────────────────

class LSTMEncoder(nn.Module):
    """Encodes a source sequence into a (hidden, cell) context vector."""
    def __init__(self, vocab_size, embed_size, hidden_size,
                 embedding_matrix=None, freeze_embeddings=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell               # each: 1 × B × H


class LSTMDecoder(nn.Module):
    """Decodes one token at a time using the encoder's context."""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm      = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        """
        x      : B × 1  (single token)
        returns: (B × 1 × V logits, hidden, cell)
        """
        x            = self.embedding(x)               # B × 1 × E
        out, (h, c)  = self.lstm(x, (hidden, cell))    # B × 1 × H
        return self.fc(out), h, c