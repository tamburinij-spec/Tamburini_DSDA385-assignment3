# Assignment 3: Natural Language Processing

## 1. Project Overview

This project explores deep learning techniques for two core Natural Language Processing tasks:

- **Task 1 – Text Generation**: Given a sequence of words from Shakespeare's works, a neural network predicts the next word(s), learning the stylistic and syntactic patterns of the text.
- **Task 2 – Machine Translation**: A sequence-to-sequence neural network translates English sentences into Spanish using the spa-eng dataset.

Both tasks are implemented with two neural network architectures (LSTM and GRU) and two embedding strategies (one-hot/random and GloVe pretrained), resulting in four experimental configurations per task. Results are compared using appropriate metrics: **perplexity** for text generation and **BLEU score** for translation.

---

## 2. Dataset Description

### Shakespeare Dataset (`data/shakespeare.txt`)
A collection of Shakespeare's works used for character- and word-level language modelling. The text is lowercased and tokenised into words. The vocabulary is capped at the 5,000 most frequent words, with unknown tokens mapped to `<unk>`.

| Property | Value |
|---|---|
| Total tokens | 204,062 |
| Vocabulary size | 5,000 |
| Sequence length | 30 |
| Task | Text generation (next-word prediction) |

### spa-eng Dataset (`data/spa-eng.txt`)
A set of English–Spanish sentence pairs sourced from the Tatoeba project. Sentences longer than 12 words are filtered out to keep training tractable. The dataset is split into training and validation sets.

| Property | Value |
|---|---|
| Total pairs loaded | *(insert value printed at runtime)* |
| Max sentence length | 12 words |
| English vocab size | *(insert value printed at runtime)* |
| Spanish vocab size | *(insert value printed at runtime)* |
| Train / Val split | 90% / 10% |
| Task | Sequence-to-sequence translation |

---

## 3. Model Architectures

### LSTM (Long Short-Term Memory)

Used for both text generation (`task1_onehot.py`, `task1_glove.py`) and as an encoder-decoder for translation (`task2_onehot.py`).

**Text generation:**  
An embedding layer maps each input token to a dense vector. A single-layer LSTM processes the sequence and a fully connected layer projects the hidden state to vocabulary logits at each timestep.

**Translation:**  
A dedicated `LSTMEncoder` compresses the source sentence into a `(hidden, cell)` context vector. A `LSTMDecoder` then generates the target sentence one token at a time, conditioned on that context and using teacher forcing during training.

### GRU (Gated Recurrent Unit)

Used for both text generation (`task1_onehot.py`, `task1_glove.py`) and as an encoder-decoder for translation (`task2_glove.py`).

**Text generation:**  
Same structure as LSTM but with GRU cells, which have fewer parameters (no separate cell state) and tend to train faster.

**Translation:**  
A `GRUEncoder` produces a single hidden context vector. A `GRUDecoder` generates the target sentence one token at a time using teacher forcing.

### Shared training details

| Hyperparameter | Task 1 | Task 2 |
|---|---|---|
| Hidden size | 256 | 256 |
| Embed size (one-hot) | 128 | 128 |
| Embed size (GloVe) | 100 | 100 |
| Epochs | 15 | 20 |
| Batch size | 64 | 64 |
| Optimiser | Adam (lr=0.001) | Adam (lr=0.001) |
| Teacher forcing ratio | — | 0.5 |
| Gradient clipping | 1.0 | 1.0 |

---

## 4. Word Embedding Methods

### One-Hot / Random Embeddings
Implemented in `embeddings/one_hot.py`. Each token is assigned a learnable dense vector initialised randomly via `nn.Embedding`. During training these vectors are updated by backpropagation, so the model learns token representations from scratch based solely on the task data. This is referred to as "one-hot" to contrast it with pretrained embeddings, even though the implementation uses a standard learnable embedding table rather than sparse one-hot vectors (which would be memory-prohibitive at scale).

### GloVe Pretrained Embeddings
Implemented in `embeddings/glove.py`. GloVe (Global Vectors for Word Representation) embeddings are loaded from `glove.6B.100d.txt`, providing 100-dimensional vectors pretrained on a large general-purpose corpus. Tokens present in the GloVe vocabulary are initialised to their pretrained vectors; tokens absent from GloVe are left as zeros. The embedding weights are frozen during training (`requires_grad = False`), meaning the model benefits from pretrained semantic knowledge without modifying it.

---

## 5. Experimental Results

### Task 1 – Text Generation (Perplexity ↓ lower is better)

Run with:
```
python task1_onehot.py | tee results_task1_onehot.txt
python task1_glove.py  | tee results_task1_glove.txt
```

| Model | Embedding | Final Loss | Perplexity |
|---|---|---|---|
| LSTM | One-Hot | 0.9689 | 2.6 |
| LSTM | GloVe | *(insert)* | *(insert)* |
| GRU | One-Hot | *(insert)* | *(insert)* |
| GRU | GloVe | *(insert)* | *(insert)* |

**Sample generated text (LSTM + One-Hot):**
```
Sample → to be or not to be the people in him and he that <unk> his receipt even so may likeness well to say and that the violent <unk> go all are gone and once again bestride our foaming steeds and once again cry <unk> upon our
```

**Sample generated text (GRU + GloVe):**
```
*(paste output of generate() from task1_glove.py here)*
```

---

### Task 2 – Machine Translation (BLEU ↑ higher is better)

Run with:
```
python task2_onehot.py | tee results_task2_onehot.txt
python task2_glove.py  | tee results_task2_glove.txt
```

| Model | Embedding | Final BLEU |
|---|---|---|
| LSTM Encoder-Decoder | One-Hot | *(insert)* |
| GRU Encoder-Decoder | GloVe | *(insert)* |

**Sample translations (GRU + GloVe):**

| English | Predicted | Ground Truth |
|---|---|---|
| *(insert)* | *(insert)* | *(insert)* |
| *(insert)* | *(insert)* | *(insert)* |
| *(insert)* | *(insert)* | *(insert)* |

---

## 6. Comparison of Models

### Architecture comparison (LSTM vs GRU)

GRUs have fewer parameters than LSTMs because they use two gates (reset and update) instead of three (input, forget, output) and have no separate cell state. In practice this means GRUs train faster per epoch and are less prone to overfitting on smaller datasets. LSTMs can in principle capture longer-range dependencies more precisely due to the cell state, which may give them an edge on longer Shakespeare sequences. Whether this advantage materialises depends on the dataset size and training duration.

*(Once you have numbers: comment here on which architecture achieved lower perplexity / higher BLEU and by how much.)*

### Embedding comparison (One-Hot vs GloVe)

One-hot/random embeddings start with no linguistic knowledge and must learn token representations entirely from the task data. GloVe embeddings arrive pretrained on a large corpus and encode semantic similarity — words with related meanings are close in vector space. This prior knowledge is especially valuable when task data is limited.

However, GloVe embeddings are frozen during training in this implementation, which means the model cannot adapt them to the specific domain (Early Modern English for Shakespeare, or short conversational sentences for spa-eng). In some cases, trainable random embeddings can outperform frozen GloVe vectors if the training set is large enough and domain-specific vocabulary is important.

*(Once you have numbers: comment here on whether GloVe embeddings improved or hurt performance, and hypothesise why.)*

---

## 7. Challenges Faced During Implementation

**Target sequence offset bug:** An early version of `utils.py` set the target sequences equal to the input sequences (`data[i:i+seq_length]`) rather than shifted by one position (`data[i+1:i+seq_length+1]`). This meant the model was trained to copy its input rather than predict the next token, resulting in misleadingly low loss values without learning anything meaningful. Identifying this required carefully re-reading the sequence creation logic.

**Inconsistent model interfaces:** The original LSTM and GRU models had different constructor signatures — the GRU accepted separate `input_vocab_size` and `output_vocab_size` arguments while the LSTM did not support GloVe at all. This made it impossible to swap architectures cleanly in a loop. Both models were unified to accept `(vocab_size, embed_size, hidden_size, embedding_matrix=None)`.

**GloVe coverage on Shakespeare vocabulary:** Shakespeare uses archaic and domain-specific vocabulary (`thou`, `hath`, `dost`) that is largely absent from the GloVe corpus, which was trained on modern web text. Many tokens therefore received zero vectors, reducing the practical advantage of pretrained embeddings for this particular dataset.

**Memory and speed trade-offs:** Processing the full Shakespeare dataset with long sequences and large batch sizes required careful tuning of `MAX_VOCAB`, `SEQ_LENGTH`, and `BATCH_SIZE` to balance training time against model quality. Gradient clipping (max norm 1.0) was added to prevent exploding gradients, which occurred early in training without it.

**Seq2seq length mismatch:** When padding source and target sequences to different lengths in Task 2, aligning the encoder output with decoder input required careful handling of the `<sos>` and `<eos>` special tokens and the `ignore_index` argument in `CrossEntropyLoss` to avoid penalising padding positions.

---

## 8. Limitations of the Considered Models

**No attention mechanism:** Both the LSTM and GRU encoder-decoders compress the entire source sentence into a single fixed-size context vector. For longer sentences this bottleneck causes the decoder to lose information about earlier tokens. Attention mechanisms solve this by allowing the decoder to query all encoder hidden states at each decoding step, but were not implemented here.

**Frozen GloVe embeddings:** Because the GloVe weights are not updated during training, the model cannot adapt them to the task domain. Fine-tunable embeddings — or using contextual embeddings from a model like BERT — would likely improve performance.

**Greedy decoding:** At inference time, both translation models select the highest-probability token at each step (greedy decoding). This is fast but suboptimal; beam search maintains multiple candidate sequences in parallel and generally yields higher BLEU scores.

**Small training sets relative to vocabulary:** With 8,000 sentence pairs for translation and a capped vocabulary, many word combinations are never seen during training. The models therefore generalise poorly to sentence structures or vocabulary that deviate from the training distribution.

**Single-layer recurrent networks:** Both LSTM and GRU models use only one recurrent layer. Stacking multiple layers typically improves representation capacity, especially for translation, where the encoder needs to build an abstract meaning representation of the source.

---

## 9. Possible Future Improvements

- **Add attention:** Implementing Bahdanau or Luong attention in the seq2seq models would significantly improve translation quality, especially for longer sentences, by letting the decoder focus on relevant source positions.
- **Transformer architecture:** Replacing the recurrent encoder-decoder with a Transformer would allow parallelised training and superior handling of long-range dependencies.
- **Beam search decoding:** Replacing greedy decoding with beam search (beam width 4–10) would improve BLEU scores at inference time without any retraining.
- **Fine-tunable pretrained models:** Using a pretrained model such as Helsinki-NLP's MarianMT (available on HuggingFace) as a baseline or starting point would dramatically raise translation quality.
- **Larger datasets:** Training on the full WMT English–Spanish corpus rather than the 8,000-pair Tatoeba subset would expose the model to far more linguistic diversity.
- **Subword tokenisation:** Replacing word-level tokenisation with byte-pair encoding (BPE) or WordPiece would reduce the unknown-token problem for rare and out-of-vocabulary words, including Shakespeare's archaic terms.
- **Hyperparameter search:** Systematically tuning hidden size, number of layers, learning rate, and dropout via a grid or random search could meaningfully improve results on both tasks.