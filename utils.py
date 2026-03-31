import re
import torch
from collections import Counter


def tokenize(text):
    """Lowercase and extract word tokens."""
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(token_lists, max_vocab=None, specials=None):
    """
    Build word ↔ index mappings from a list of token lists.

    Args:
        token_lists : list of lists of strings
        max_vocab   : cap on non-special vocabulary size
        specials    : special tokens prepended to the vocab (e.g. ['<pad>', '<sos>'])

    Returns:
        vocab    : list of tokens in index order
        word2idx : dict  token → index
        idx2word : dict  index → token
    """
    specials = specials or []
    counter  = Counter(tok for tokens in token_lists for tok in tokens)
    words    = [w for w, _ in counter.most_common(max_vocab)]
    vocab    = specials + words
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word


def create_sequences(data, seq_length):
    """
    Slide a window over *data* to create (input, target) pairs.

    Target is the input shifted **right by one position** so the model
    learns to predict the next token at every step.

    Args:
        data       : 1-D LongTensor of encoded tokens
        seq_length : context window size

    Returns:
        X : (N, seq_length) — input sequences
        y : (N, seq_length) — targets (each y[i] == data[i+1 : i+seq_length+1])
    """
    X = torch.stack([data[i    : i + seq_length    ] for i in range(len(data) - seq_length)])
    y = torch.stack([data[i + 1: i + seq_length + 1] for i in range(len(data) - seq_length)])
    return X, y


def encode(tokens, word2idx, unk_idx=0):
    """Map a list of tokens to indices, replacing unknowns with unk_idx."""
    return [word2idx.get(t, unk_idx) for t in tokens]