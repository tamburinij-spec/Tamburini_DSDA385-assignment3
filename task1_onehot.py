"""
Task 1 – Text Generation  |  One-Hot (random) Embeddings
=========================================================
Trains and compares two architectures on the Shakespeare dataset
using randomly initialised (learnable) embeddings:
  - LSTM + One-Hot
  - GRU  + One-Hot

Metric: Perplexity (lower is better)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import LSTMModel
from models.gru_model  import GRUModel
from embeddings.one_hot import tokenize, build_vocab
from utils import create_sequences

# ── Device ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH   = "data/shakespeare.txt"
SEQ_LENGTH  = 30
EMBED_SIZE  = 128      # free to tune — not tied to GloVe dimensions
HIDDEN_SIZE = 256
EPOCHS      = 15
BATCH_SIZE  = 64
MAX_VOCAB   = 5000

# ── Load & tokenise ───────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

tokens = tokenize(raw)
print(f"Total tokens : {len(tokens):,}")

vocab, word2idx, idx2word = build_vocab([tokens], max_vocab=MAX_VOCAB - len(["<pad>","<sos>","<eos>","<unk>"]))
vocab_size = len(vocab)
print(f"Vocab size   : {vocab_size:,}")

encoded = torch.tensor(
    [word2idx.get(t, word2idx["<unk>"]) for t in tokens], dtype=torch.long
)

# ── Sequences & DataLoader ────────────────────────────────────────────
X, y    = create_sequences(encoded, SEQ_LENGTH)
loader  = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
print(f"Sequences    : {len(X):,}")

# ── Training ──────────────────────────────────────────────────────────
def train(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)                        # B × T × V
            loss   = criterion(
                output.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        final_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:>2}/{epochs}  "
                  f"loss={final_loss:.4f}  perplexity={math.exp(final_loss):.1f}")

    return final_loss

# ── Text generation ───────────────────────────────────────────────────
def generate(model, prompt, length=40, temperature=0.8):
    model.eval()
    unk = word2idx["<unk>"]
    ids = [word2idx.get(w, unk) for w in tokenize(prompt)]
    if len(ids) < SEQ_LENGTH:
        ids = [0] * (SEQ_LENGTH - len(ids)) + ids

    seq    = torch.tensor(ids[-SEQ_LENGTH:]).unsqueeze(0).to(device)
    result = tokenize(prompt)

    with torch.no_grad():
        for _ in range(length):
            out  = model(seq)
            prob = F.softmax(out[:, -1, :] / temperature, dim=-1)
            pred = torch.multinomial(prob, 1).item()
            result.append(idx2word.get(pred, "<unk>"))
            seq  = torch.cat(
                [seq[:, 1:], torch.tensor([[pred]]).to(device)], dim=1
            )
    return " ".join(result)

# ── Run experiments ───────────────────────────────────────────────────
experiments = [
    ("LSTM + One-Hot", LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)),
    ("GRU  + One-Hot", GRUModel (vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)),
]

results = {}
for name, model in experiments:
    print(f"\n{'='*55}\n  Training: {name}\n{'='*55}")
    loss       = train(model, loader, EPOCHS)
    ppl        = math.exp(loss)
    results[name] = ppl
    print(f"  Sample → {generate(model, 'to be or not to be')}")

# ── Comparison table ──────────────────────────────────────────────────
print("\n\n" + "="*42)
print(f"{'Model':<22} {'Perplexity':>12}")
print("-"*35)
for name, ppl in results.items():
    print(f"{name:<22} {ppl:>12.2f}")
print("="*42)
print("(Lower perplexity = better)")