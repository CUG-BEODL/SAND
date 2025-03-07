


# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm


class config():
    """
    Configuration class for model parameters and dataset settings.
    """

    def __init__(self):
        self.area = 'Xiongan_New_Area'  # Target study area
        self.device = 'cuda'  # Use CUDA if available, otherwise fallback to CPU
        self.input_channels = 10  # Number of input data channels
        self.num_classes = 6  # Number of output classification categories
        self.src_vocab_size = self.input_channels  # Vocabulary size for source data
        self.tgt_vocab_size = self.num_classes + 3  # Target vocabulary (+ start, end, padding tokens)

        # Model hyperparameters
        self.epochs = 201  # Number of training epochs
        self.num_layers = 4  # Number of transformer layers
        self.num_heads = 6 # Number of attention heads
        self.hidden_dim = 192  # Dimension of the model embeddings
        self.feedforward_dim = 256  # Dimension of the feedforward layer
        self.dropout_rate = 0.1  # Dropout rate for regularization
        self.max_seq_length = 5  # Maximum sequence length
        self.batch_size = 64  # Batch size for training

        self.selected_months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Selected months for processing

        # Special tokens
        self.bos_token = 0  # Beginning of Sequence (BOS)
        self.eos_token = self.num_classes + 1  # End of Sequence (EOS)
        self.pad_token = self.num_classes + 2  # Padding token (PAD)

        # Model save path
        self.save_path = os.path.join('SaveModel', f'{self.area}_len_{self.selected_months.shape[0]}.pt')
