import torch

def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i:i+seq_length])

    return torch.tensor(sequences), torch.tensor(targets)

# Build vocab for source and target
def build_vocab(sentences):
    vocab = {}
    idx = 0
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab