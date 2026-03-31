import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU language model for text generation.
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

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x      = self.embedding(x)        # B × T → B × T × E
        out, _ = self.gru(x)              # B × T × H
        return self.fc(out)               # B × T × V


# ── Seq2Seq components for Task 2 ─────────────────────────────────────

class GRUEncoder(nn.Module):
    """Encodes a source sequence into a hidden context vector."""
    def __init__(self, vocab_size, embed_size, hidden_size,
                 embedding_matrix=None, freeze_embeddings=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x       = self.embedding(x)
        _, hidden = self.gru(x)
        return hidden                     # 1 × B × H


class GRUDecoder(nn.Module):
    """Decodes one token at a time using the encoder's context."""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru       = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        """
        x      : B × 1  (single token)
        returns: (B × 1 × V logits, hidden)
        """
        x          = self.embedding(x)         # B × 1 × E
        out, hidden = self.gru(x, hidden)      # B × 1 × H
        return self.fc(out), hidden