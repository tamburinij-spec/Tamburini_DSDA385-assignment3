import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from models.gru_model import GRUModel
from embeddings.glove import load_glove_embeddings

# ------------------------
# Example dataset (English → Spanish)
# ------------------------
pairs = [
    ("hello", "hola"),
    ("how are you", "como estas"),
    ("i am fine", "estoy bien"),
    ("good morning", "buenos dias"),
    ("thank you", "gracias")
]

# ------------------------
# Build vocabularies
# ------------------------
def build_vocab(sentences):
    vocab = {}
    idx = 0
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

eng_vocab = build_vocab([p[0] for p in pairs])
spa_vocab = build_vocab([p[1] for p in pairs])

def encode(sentence, vocab):
    return [vocab[w] for w in sentence.split()]

# ------------------------
# Encode and pad sequences
# ------------------------
X = [torch.tensor(encode(p[0], eng_vocab)) for p in pairs]
y = [torch.tensor(encode(p[1], spa_vocab)) for p in pairs]

X = pad_sequence(X, batch_first=True, padding_value=0)
y = pad_sequence(y, batch_first=True, padding_value=0)

# Make input and target same length for simplicity
max_len = max(X.shape[1], y.shape[1])
def pad_to_len(tensor, length):
    if tensor.shape[1] < length:
        pad_size = length - tensor.shape[1]
        pad = torch.zeros((tensor.shape[0], pad_size), dtype=torch.long)
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor

X = pad_to_len(X, max_len)
y = pad_to_len(y, max_len)

# ------------------------
# Load GloVe embeddings
# ------------------------
embedding_matrix = load_glove_embeddings(
    "embeddings/glove.6B.100d.txt",
    eng_vocab,
    embed_size=100
)

# ------------------------
# Initialize model
# ------------------------
model = GRUModel(
    input_vocab_size=len(eng_vocab),
    output_vocab_size=len(spa_vocab),
    embed_size=100,
    hidden_size=128,
    embedding_matrix=embedding_matrix
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# Training loop
# ------------------------
batch_size = 2  # small for demo
num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0

    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        optimizer.zero_grad()
        output = model(x_batch)  # shape: batch × seq × vocab

        loss = criterion(
            output.reshape(-1, len(spa_vocab)),
            y_batch.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ------------------------
# Test generation (greedy)
# ------------------------

idx2word = {i: w for w, i in spa_vocab.items()}

def translate(src_sentence, max_len=10):
    model.eval()

    # Encode source sentence
    input_seq = torch.tensor([encode(src_sentence, eng_vocab)]).long()

    result_indices = []

    for _ in range(max_len):
        output = model(input_seq)
        # pick last timestep for next word
        pred_idx = torch.argmax(output[:, -1, :], dim=1).item()
        result_indices.append(pred_idx)

        # append prediction to input for next timestep
        input_seq = torch.cat([input_seq, torch.tensor([[pred_idx]])], dim=1)

    # Convert indices to words
    translation = [idx2word.get(idx, "<unk>") for idx in result_indices]
    return " ".join(translation)

# Example
for src, tgt in pairs:
    print(f"{src} -> {translate(src)} (expected: {tgt})")