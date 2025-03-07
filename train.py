import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from dataloader import load_data, Batch
from Config import config
from utils import *
from test import evaluate

# Load configuration parameters
config = config()
EPOCHS = config.epochs  # Number of training epochs
EMBEDDING_DIM = config.hidden_dim  # Embedding dimension
DEVICE = config.device
PAD_TOKEN = config.pad_token  # Padding token index
SAVE_PATH = config.save_path

class Criterion(nn.Module):
    """
    Implements label smoothing to reduce overconfidence in predictions.
    """

    def __init__(self, padding_idx):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=padding_idx)

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


class SimpleLossCompute:
    """
    Computes the loss and updates model parameters.
    """

    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, predictions, targets, norm_factor):
        predictions = self.generator(predictions)
        loss = self.criterion(predictions.contiguous().view(-1, predictions.size(-1)),
                              targets.contiguous().view(-1)) / norm_factor

        loss.backward()
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.item() * norm_factor.float()


class NoamOpt:
    """
    Optimizer wrapper implementing learning rate scheduling.
    """

    def __init__(self, model_size, factor, warmup_steps, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """ Update learning rate and perform optimization step. """
        self._step += 1
        rate = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def get_lr(self, step=None):
        """ Calculate the learning rate schedule. """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)


def run_epoch(data_loader, model, loss_fn, epoch):
    """ Run one epoch of training or evaluation. """
    start_time = time.time()
    total_tokens, total_loss, tokens_processed = 0, 0, 0

    for i, (src_data, trg_data) in enumerate(data_loader):
        batch = Batch(src_data, trg_data)
        src, trg, src_mask, trg_mask = (
            batch.src.to(DEVICE).float(),
            batch.trg.to(DEVICE).long(),
            batch.src_mask.to(DEVICE).bool(),
            batch.trg_mask.to(DEVICE).bool()
        )
        trg_y, num_tokens = batch.trg_y.to(DEVICE).long(), batch.ntokens.to(DEVICE).long()

        # Forward pass
        output = model(src, trg, src_mask, trg_mask)
        loss = loss_fn(output, trg_y, num_tokens)

        total_loss += loss
        total_tokens += num_tokens
        tokens_processed += num_tokens

        if i % 50 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch} Batch {i}: Loss = {loss / num_tokens:.4f}, Tokens/sec = {tokens_processed.float() / elapsed_time / 1000.:.2f}K")
            start_time = time.time()
            tokens_processed = 0

    return total_loss / total_tokens


def train_model(data, model, criterion, optimizer):
    """ Train the model with given data and optimizer. """
    best_score = -1000
    train_data, val_data, _ = data

    for epoch in range(EPOCHS):
        model.train()
        run_epoch(train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

        model.eval()
        print('>>>>> Evaluating on validation set...')
        # dev_loss = run_epoch(val_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        score = evaluate(val_data, model)
        print(f'<<<<< Validation score: {score:.4f}')

        # Save the best model
        if score > best_score:
            torch.save(model.state_dict(), SAVE_PATH.replace('.pt', f'_{round(score, 2)}.pt'))
            best_score = score
            print('****** Model saved ******')
        print()


if __name__ == '__main__':


    # Load dataset
    dataset = load_data(config.area)
    src_vocab_size, tgt_vocab_size = config.src_vocab_size, config.tgt_vocab_size
    print(f"Source Vocab: {src_vocab_size}, Target Vocab: {tgt_vocab_size}")

    # Initialize model
    model = built(config)
    print(">>>>>>> Training started")
    start_time = time.time()

    criterion = Criterion(padding_idx=PAD_TOKEN)
    optimizer = NoamOpt(EMBEDDING_DIM, 1, 500, optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train_model(dataset, model, criterion, optimizer)
    print(f"<<<<<<< Training finished in {time.time() - start_time:.2f} seconds")
