"""
Task 2 – Machine Translation  |  One-Hot (random) Embeddings
=============================================================
Trains an LSTM encoder-decoder on the spa-eng.txt dataset
using randomly initialised (learnable) embeddings.

Metric: BLEU score (higher is better)
"""

import re
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from models.lstm_model import LSTMEncoder, LSTMDecoder
from embeddings.one_hot import tokenize, build_vocab

# ── Device ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH       = "data/spa-eng.txt"
MAX_PAIRS       = 8000
MAX_LEN         = 12
EMBED_SIZE      = 128     # free to tune
HIDDEN_SIZE     = 256
EPOCHS          = 20
BATCH_SIZE      = 64
TEACHER_FORCING = 0.5
TRAIN_SPLIT     = 0.9

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

# ── Load data ─────────────────────────────────────────────────────────
def load_pairs(path, max_pairs):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) >= 2:
                pairs.append((tokenize(cols[0]), tokenize(cols[1])))
    pairs = [
        (e, s) for e, s in pairs
        if 1 <= len(e) <= MAX_LEN and 1 <= len(s) <= MAX_LEN
    ]
    return pairs[:max_pairs]

pairs = load_pairs(DATA_PATH, MAX_PAIRS)
random.shuffle(pairs)
print(f"Loaded {len(pairs):,} sentence pairs")

eng_vocab, eng_w2i, eng_i2w = build_vocab([e for e, _ in pairs])
spa_vocab, spa_w2i, spa_i2w = build_vocab([s for _, s in pairs])
print(f"English vocab: {len(eng_vocab):,}  |  Spanish vocab: {len(spa_vocab):,}")

# ── Encode & pad ──────────────────────────────────────────────────────
def encode_seq(tokens, w2i):
    unk = w2i[UNK]
    return [SOS_IDX] + [w2i.get(t, unk) for t in tokens] + [EOS_IDX]

def pairs_to_tensors(pair_list):
    src = [torch.tensor(encode_seq(e, eng_w2i), dtype=torch.long) for e, _ in pair_list]
    tgt = [torch.tensor(encode_seq(s, spa_w2i), dtype=torch.long) for _, s in pair_list]
    return (pad_sequence(src, batch_first=True, padding_value=PAD_IDX),
            pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX))

split = int(TRAIN_SPLIT * len(pairs))
train_src, train_tgt = pairs_to_tensors(pairs[:split])
val_src,   val_tgt   = pairs_to_tensors(pairs[split:])

train_loader = DataLoader(TensorDataset(train_src, train_tgt), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_src,   val_tgt),   batch_size=BATCH_SIZE)
print(f"Train batches: {len(train_loader)}  |  Val pairs: {len(pairs) - split}")

# ── Seq2Seq (LSTM) ────────────────────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, tgt_len = tgt.shape
        tgt_vocab  = self.decoder.fc.out_features
        outputs    = torch.zeros(B, tgt_len, tgt_vocab).to(device)

        hidden, cell  = self.encoder(src)
        dec_input     = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            out, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t, :] = out.squeeze(1)
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input   = tgt[:, t].unsqueeze(1) if use_teacher else out.argmax(2)

        return outputs

eng_sz = len(eng_vocab)
spa_sz = len(spa_vocab)

model = Seq2SeqLSTM(
    LSTMEncoder(eng_sz, EMBED_SIZE, HIDDEN_SIZE).to(device),
    LSTMDecoder(spa_sz, EMBED_SIZE, HIDDEN_SIZE).to(device),
).to(device)

# ── Training & BLEU ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        out  = model(src, tgt, TEACHER_FORCING)
        loss = criterion(out[:, 1:, :].reshape(-1, spa_sz), tgt[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate_bleu(model, loader):
    model.eval()
    refs, hyps = [], []
    sf = SmoothingFunction().method1
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out   = model(src, tgt, teacher_forcing_ratio=0.0)
            preds = out.argmax(2)
            for i in range(src.size(0)):
                refs.append([[spa_i2w[x.item()] for x in tgt[i]
                               if x.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]])
                hyps.append( [spa_i2w[x.item()] for x in preds[i]
                               if x.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)])
    return corpus_bleu(refs, hyps, smoothing_function=sf)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print(f"\n{'='*55}\n  Training: LSTM + One-Hot\n{'='*55}")
for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader, optimizer, criterion)
    if (epoch + 1) % 5 == 0:
        bleu = evaluate_bleu(model, val_loader)
        print(f"    Epoch {epoch+1:>2}/{EPOCHS}  loss={loss:.4f}  BLEU={bleu:.4f}")

final_bleu = evaluate_bleu(model, val_loader)
print(f"\nFinal BLEU (LSTM + One-Hot): {final_bleu:.4f}")

# ── Sample translations ───────────────────────────────────────────────
def translate(model, src_tokens, max_len=20):
    model.eval()
    src = torch.tensor([encode_seq(src_tokens, eng_w2i)], dtype=torch.long).to(device)
    tgt = torch.zeros(1, max_len, dtype=torch.long).to(device)
    tgt[0, 0] = SOS_IDX
    with torch.no_grad():
        out = model(src, tgt, teacher_forcing_ratio=0.0)
    words = []
    for idx in out.argmax(2)[0]:
        w = spa_i2w[idx.item()]
        if w == EOS:
            break
        if w not in (SOS, PAD, UNK):
            words.append(w)
    return " ".join(words)

val_pairs = pairs[split:]
print("\n── Sample translations ──────────────────────────────────────")
for eng, spa in val_pairs[:5]:
    print(f"  EN:   {' '.join(eng)}")
    print(f"  Pred: {translate(model, eng)}")
    print(f"  True: {' '.join(spa)}\n")