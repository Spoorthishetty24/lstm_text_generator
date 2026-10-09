"""
LSTM Text Generator (word-level) trained on Shakespeare
=======================================================

Pipeline:
  1. Load the text, lowercase it, remove punctuation
  2. Turn words into numbers (tokenize)
  3. Build (previous words -> next word) training pairs
  4. Build an Embedding -> LSTM -> Dense(softmax) model in Keras
  5. Train with a train/validation split, early stopping and checkpoints
  6. Generate new text from seed sentences

Dataset (tiny Shakespeare, ~1.1 MB, from Andrej Karpathy's char-rnn repo):
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  (Full works also at https://www.gutenberg.org/ebooks/100)

Example:
  python lstm_text_generator.py --tag baseline --seq_len 5 --layers 1 --units 128
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep the console tidy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------------------------------------------------------
# 0. Settings you can change from the command line
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="shakespeare.txt")
parser.add_argument("--tag", default="baseline", help="name for this experiment")
parser.add_argument("--seq_len", type=int, default=5, help="how many words the model looks at")
parser.add_argument("--layers", type=int, default=1, help="number of stacked LSTM layers")
parser.add_argument("--units", type=int, default=128, help="memory cells per LSTM layer")
parser.add_argument("--embed_dim", type=int, default=100)
parser.add_argument("--vocab_size", type=int, default=5000, help="keep only the most common words")
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

tf.keras.utils.set_random_seed(args.seed)
OUT = f"outputs/{args.tag}"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load + clean the text
# ----------------------------------------------------------------------------
with open(args.data, encoding="utf-8") as f:
    raw_text = f.read()

def clean(t):
    """lowercase, drop apostrophes (king's -> kings), turn other punctuation into spaces"""
    t = t.lower().replace("'", "")
    return re.sub(r"[^a-z\s]", " ", t)

words = clean(raw_text).split()               # split into word tokens
print(f"Total words in the book: {len(words):,}")

# ----------------------------------------------------------------------------
# 2. Tokenize: give every word a number
#    Rare words become <unk> so the output layer stays a manageable size.
# ----------------------------------------------------------------------------
from collections import Counter
counts = Counter(words)
most_common = [w for w, _ in counts.most_common(args.vocab_size - 1)]
index_to_word = ["<unk>"] + most_common       # index 0 = unknown word
word_to_index = {w: i for i, w in enumerate(index_to_word)}
vocab_size = len(index_to_word)
ids = np.array([word_to_index.get(w, 0) for w in words], dtype=np.int32)
print(f"Vocabulary size: {vocab_size:,}  (unknown-word share: {(ids == 0).mean():.1%})")

# ----------------------------------------------------------------------------
# 3. Input/output pairs: `seq_len` words in -> the next word out
# ----------------------------------------------------------------------------
n = len(ids) - args.seq_len
X = np.stack([ids[i:i + n] for i in range(args.seq_len)], axis=1)  # shape (n, seq_len)
y = ids[args.seq_len:]                                              # shape (n,)
print(f"Training pairs: {len(X):,}   input shape: {X.shape}")

# Split: first 90% train, last 10% validation (kept in book order, no leakage by shuffling)
split = int(0.9 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# ----------------------------------------------------------------------------
# 4. The model: Embedding -> LSTM(s) -> Dense softmax
# ----------------------------------------------------------------------------
model = models.Sequential(name=args.tag)
model.add(layers.Input(shape=(args.seq_len,)))
model.add(layers.Embedding(vocab_size, args.embed_dim))           # word number -> vector of meaning
for i in range(args.layers):
    last = (i == args.layers - 1)
    model.add(layers.LSTM(args.units, return_sequences=not last))  # memory layer(s)
    model.add(layers.Dropout(0.2))                                  # randomly forget a bit -> less cheating
model.add(layers.Dense(vocab_size, activation="softmax"))          # probability for every word

# sparse_categorical_crossentropy == categorical crossentropy, but takes the word
# number directly instead of a giant one-hot vector (saves a lot of memory).
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=["accuracy"])
model.summary()

# ----------------------------------------------------------------------------
# 5. Train, with early stopping + checkpoint
# ----------------------------------------------------------------------------
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    callbacks.ModelCheckpoint(f"{OUT}/best.keras", monitor="val_loss", save_best_only=True),
]
t0 = time.time()
hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=args.epochs, batch_size=args.batch_size, callbacks=cbs, verbose=2)
train_minutes = (time.time() - t0) / 60

val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=1024, verbose=0)
perplexity = float(np.exp(val_loss))
print(f"\nBest validation loss {val_loss:.3f} | accuracy {val_acc:.1%} | perplexity {perplexity:.0f}")

# ----------------------------------------------------------------------------
# 6. Text generation
#    temperature < 1 : safe, repetitive    temperature > 1 : wild, creative
# ----------------------------------------------------------------------------
def generate(seed_text, n_words=40, temperature=0.8):
    seed = clean(seed_text).split()
    out = list(seed)
    for _ in range(n_words):
        window = [word_to_index.get(w, 0) for w in out[-args.seq_len:]]
        window = [0] * (args.seq_len - len(window)) + window         # pad if seed is short
        probs = model.predict(np.array([window]), verbose=0)[0].astype("float64")
        probs[0] = 0                                                  # never output <unk>
        probs = np.log(probs + 1e-9) / temperature                    # apply temperature
        probs = np.exp(probs - probs.max())
        probs /= probs.sum()
        out.append(index_to_word[np.random.choice(vocab_size, p=probs)])
    return " ".join(out)

seeds = [
    " ".join(words[:args.seq_len]),            # first words of the dataset
    "to be or not to be",
    "my lord the king",
    "romeo romeo where art thou",
]
samples = []
np.random.seed(args.seed)
for s in seeds:
    for temp in (0.6, 1.0):
        g = generate(s, n_words=40, temperature=temp)
        samples.append({"seed": s, "temperature": temp, "text": g})
        print(f"\n[seed: '{s}' | temperature {temp}]\n{g}")

# ----------------------------------------------------------------------------
# 7. Save results for the report
# ----------------------------------------------------------------------------
result = {
    "tag": args.tag, "seq_len": args.seq_len, "layers": args.layers, "units": args.units,
    "params": int(model.count_params()), "epochs_run": len(hist.history["loss"]),
    "val_loss": float(val_loss), "val_acc": float(val_acc), "perplexity": perplexity,
    "train_minutes": round(train_minutes, 1), "samples": samples,
    "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
}
with open(f"{OUT}/result.json", "w") as f:
    json.dump(result, f, indent=2)
