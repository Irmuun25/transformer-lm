#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

from transformer_lm import train_lm, NeuralLanguageModel
from utils import Indexer

# Load data
print("Loading data...", flush=True)
train_text = open('data/text8-100k.txt').read()
dev_text = open('data/text8-dev.txt').read()

print(f"Train text length: {len(train_text)}", flush=True)
print(f"Dev text length: {len(dev_text)}", flush=True)

# Create vocab
vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
vocab_index = Indexer()
for char in vocab:
    vocab_index.add_and_get_index(char)

print(f"Vocab size: {len(vocab_index)}", flush=True)

# Create test args
class Args:
    pass

args = Args()

print("Starting training...", flush=True)
model = train_lm(args, train_text, dev_text, vocab_index)
print("Training completed!", flush=True)

# Test the model
print("\nTesting model...", flush=True)
log_probs = model.get_next_char_log_probs("hello")
print(f"Log probs for 'hello': {log_probs[:5]}", flush=True)

log_prob_seq = model.get_log_prob_sequence("world", "hello ")
print(f"Log prob for 'world' after 'hello ': {log_prob_seq}", flush=True)

print("Done!", flush=True)
