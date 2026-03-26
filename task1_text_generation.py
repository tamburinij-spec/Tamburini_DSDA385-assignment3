import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import psutil
import math
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import LSTMModel
from embeddings.one_hot import build_vocab


# ------------------------
# Device (GPU if available)
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------
# Adaptive batch size
# ------------------------
def get_batch_size():
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB

    base_batch = 64  # baseline for ~16GB

    if available_ram >= 32:
        return min(512, base_batch * 4)
    elif available_ram >= 16:
        return min(256, base_batch * 2)
    elif available_ram >= 8:
        return base_batch
    else:
        return 32


# ------------------------
# Load data
# ------------------------
with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

vocab, char2idx, idx2char = build_vocab(text)
encoded = torch.tensor([char2idx[c] for c in text])


# ------------------------
# Create sequences
# ------------------------
seq_length = 40

def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+1:i+seq_length+1])

    return torch.stack(X), torch.stack(y)


X, y = create_sequences(encoded, seq_length)


# ------------------------
# DataLoader
# ------------------------
batch_size = get_batch_size()
print(f"Using adaptive batch size: {batch_size}")

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ------------------------
# Model
# ------------------------
model = LSTMModel(len(vocab), 128, 256).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ------------------------
# Training loop
# ------------------------
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        output = model(x_batch)

        loss = criterion(
            output.reshape(-1, len(vocab)),
            y_batch.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# ------------------------
# Perplexity
# ------------------------
perplexity = math.exp(avg_loss)
print(f"Perplexity: {perplexity:.2f}")


# ------------------------
# Temperature Sampling
# ------------------------
def sample(output, temperature=0.8):
    probs = F.softmax(output / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


# ------------------------
# Generate text
# ------------------------
def generate(model, start_text, length=200, temperature=0.8):
    model.eval()

    input_seq = torch.tensor(
        [char2idx[c] for c in start_text]
    ).unsqueeze(0).to(device)

    result = start_text

    for _ in range(length):
        output = model(input_seq)

        pred = sample(output[:, -1, :], temperature)
        result += idx2char[pred]

        input_seq = torch.cat(
            [input_seq[:, 1:], torch.tensor([[pred]]).to(device)],
            dim=1
        )

    return result


# ------------------------
# Run generation
# ------------------------
print("\nGenerated Text:\n")
print(generate(model, "to be or not to be", temperature=0.8))