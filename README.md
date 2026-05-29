### `README.md`

```markdown
# Transformer Language Model & Letter Counting Classifier

A PyTorch implementation of a custom Transformer network applied to character-level neural language modeling and sequence classification tasks. 

**Author**: Munkh-Irmuun Munkhbat

## Overview
This repository contains a from-scratch implementation of Transformer layers (including self-attention and positional encoding) to solve two distinct natural language processing problems:
1. **Neural Language Modeling**: Predicting the next character in a sequence using a causal-masked Transformer encoder, trained on the `text8` dataset.
2. **Letter Counting Sequence Classification**: A sequence labeling task where the model predicts whether a character has appeared 0, 1, or 2+ times previously in a 20-character sequence.

## Repository Structure

* `transformer.py`: Contains the core `TransformerLayer`, `PositionalEncoding`, and the sequence classification `Transformer` model.
* `transformer_lm.py`: Implements the `NeuralLanguageModelModule` with causal masking, and the training loop for the language model.
* `lm.py`: Main executable for the language modeling task. Includes uniform baseline models and evaluation metrics (perplexity, average log probability, and normalization checks).
* `letter_counting.py`: Main executable for the sequence classification task.
* `utils.py`: Contains data structure utilities, including an `Indexer` for vocabulary mapping and a `Beam` class for maintaining top-n elements.

## Installation

1. Clone the repository:
```bash
   git clone [https://github.com/YOUR_USERNAME/transformer-lm.git](https://github.com/YOUR_USERNAME/transformer-lm.git)
   cd transformer-lm

```

2. Install the dependencies:

```bash
   pip install -r requirements.txt

```

## Usage

### 1. Neural Language Model

Train and evaluate the causal language model on the `text8` dataset. The script will automatically run sanity checks and calculate perplexity.

```bash
python lm.py --model NEURAL --train_path data/text8-100k.txt --dev_path data/text8-dev.txt

```

### 2. Letter Counting Task

Train the transformer to classify historical character frequencies. By default, it runs the `BEFORE` task (counting only previous occurrences).

```bash
python letter_counting.py --task BEFORE --train data/lettercounting-train.txt --dev data/lettercounting-dev.txt

```

*Note: When testing the letter counting model, the system will output predicted labels and optionally generate attention map plots for the validation set*.

```

---

### `.gitignore`

This file ensures you do not accidentally commit large datasets, PyTorch model checkpoints, or compiled python files to your repository.

```text
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# PyTorch artifacts
*.pth
*.pt
*.ckpt

# Output files and plots
plots/
*.json
classifier-output.json
output.json

# Datasets
data/
*.txt

# Virtual Environments
venv/
env/
.env/

# IDE settings
.vscode/
.idea/

```

---

### `requirements.txt`

These are the required dependencies based on your imports.

```text
numpy
torch
scipy
torchvision
matplotlib

```