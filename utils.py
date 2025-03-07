from model.transformer import make_model as transformer
import numpy as np
import os
from glob import glob
from tqdm import tqdm


def built(config):
    model = transformer
    return model(config.src_vocab_size, config.tgt_vocab_size, config.num_layers, config.hidden_dim, config.feedforward_dim, config.num_heads,
                 config.dropout_rate)


def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
