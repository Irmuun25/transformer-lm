# transformer_lm.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class NeuralLanguageModelModule(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, num_positions=1000, batched=True)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
            )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, indices):
        x = self.embed(indices)
        self.pos_enc.batched = (len(indices.shape) == 2)
        x = self.pos_enc(x)

        seq_len = x.shape[1] if len(indices.shape) == 2 else x.shape[0]
        mask = self.generate_causal_mask(seq_len).to(x.device)

        output = self.transformer(x, mask=mask, is_causal=True)
        logits = self.out_proj(output)
        return self.log_softmax(logits)

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
        self.model.eval()
        self.max_context = 50

    def get_next_char_log_probs(self, context):
        self.model.eval()
        if context == "":
            context = " "
        
        if len(context) > self.max_context:
            context = context[-self.max_context:]
            
        indices = [self.vocab_index.index_of(c) for c in context]
        indices_tensor = torch.LongTensor(indices).unsqueeze(0)

        with torch.no_grad():
            log_probs = self.model(indices_tensor)

        return log_probs[0, -1, :].numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        if context == "":
            context = " "
        
        full_seq = context + next_chars
        total_log_prob = 0.0
        cxt_len = len(context)

        for i, char in enumerate(next_chars):
            avail_ctx = full_seq[:cxt_len + i]
            if len(avail_ctx) > self.max_context:
                avail_ctx = avail_ctx[-self.max_context:]

            indices = [self.vocab_index.index_of(c) for c in avail_ctx]
            indices_tensor = torch.LongTensor(indices).unsqueeze(0)

            with torch.no_grad():
                log_probs = self.model(indices_tensor)

            target_idx = self.vocab_index.index_of(char)
            prob = log_probs[0, -1, target_idx].item()
            total_log_prob += prob

        return total_log_prob  

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    vocab_size = len(vocab_index)
    d_model = 64
    nhead = 2
    num_layers = 2
    dim_feedforward = 128

    model = NeuralLanguageModelModule(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    chunk_size = 50
    batch_size = 32
    num_epochs = 20

    train_text = " " + train_text
    indices = [vocab_index.index_of(c) for c in train_text]

    chunked_inputs = []
    chunked_targets = []

    step_size = 25
    for i in range(0, len(indices)-chunk_size-1, step_size):
        chunk = indices[i:i+chunk_size+1]
        chunked_inputs.append(chunk[:-1])
        chunked_targets.append(chunk[1:])

    for t in range(num_epochs):
        model.train()
        combined = list(zip(chunked_inputs, chunked_targets))
        random.shuffle(combined)
        chunked_inputs, chunked_targets = zip(*combined)

        loss_this_epoch = 0.0
        for b in range(0, len(chunked_inputs), batch_size):
            batch_inputs = torch.LongTensor(chunked_inputs[b:b+batch_size])
            batch_targets = torch.LongTensor(chunked_targets[b:b+batch_size])

            optimizer.zero_grad()
            log_probs = model(batch_inputs)

            loss = loss_fcn(log_probs.view(-1, vocab_size), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print(f"Epoch {t+1} / {num_epochs} Loss: {loss_this_epoch:.4f}")    

    model.eval()
    return NeuralLanguageModel(model, vocab_index)