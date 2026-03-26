import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size, embedding_matrix=None):
        super(GRUModel, self).__init__()

        # Input embedding layer
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False  # freeze GloVe embeddings

        # GRU layer
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        # Output layer (to target vocab size)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out