#!/usr/bin/env python
import torch
import numpy as np
from transformer_lm import CausalTransformerLM, NeuralLanguageModel
from utils import Indexer

# Create a small test
vocab_size = 27
d_model = 64
d_internal = 64
num_layers = 2

print("Creating model...")
model = CausalTransformerLM(vocab_size, d_model, d_internal, num_layers, max_seq_len=100)
model.eval()

print("Model created successfully!")

# Test forward pass
test_input = torch.LongTensor([0, 1, 2, 3, 4])  # Simple sequence
print(f"Test input shape: {test_input.shape}")

with torch.no_grad():
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output log probs (first position):\n{output[0]}")
    
print("Model test passed!")
