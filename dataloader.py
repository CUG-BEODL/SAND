#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.autograd import Variable
from Config import config
from utils import *
from torch.utils import data

# Load configuration parameters
config = config()
PAD_TOKEN = config.pad_token  # Padding token index
BOS_TOKEN = config.bos_token  # Begin of sequence
EOS_TOKEN = config.eos_token  # End of sequence
BATCH_SIZE = config.batch_size  # Batch size
DEVICE = config.device  # Computation device (CPU/GPU)


def subsequent_mask(size):
    """
    Generate a mask to hide future words for sequence processing.
    :param size: Sequence length
    :return: Mask tensor (upper triangular matrix with False in upper part, True in lower part)
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Convert to boolean tensor


class Batch:
    """
    Batch object for handling input sequences and corresponding masks.
    """

    def __init__(self, src, trg=None, pad=PAD_TOKEN):
        self.src = src.float()
        self.src_mask = (src != pad)[:, :, 0].unsqueeze(-2)  # Mask for non-padding elements

        if trg is not None:
            self.trg = trg[:, :-1]  # Target input (excluding last token)
            self.trg_y = trg[:, 1:]  # Target output (excluding first token)
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()  # Number of valid tokens

    @staticmethod
    def make_std_mask(tgt, pad):
        """ Create standard mask for target sequence """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MaskDataset(data.Dataset):
    """
    Custom Dataset class for handling sequence data with mask processing.
    """

    def __init__(self, dataset, label):
        super(MaskDataset, self).__init__()
        self.dataset = dataset
        self.label = label

    def __getitem__(self, index):
        data, label = self.dataset[index], self.label[index]
        num_channels = data.shape[0]

        # Append BOS and EOS tokens to the sequence
        start_token = np.tile(np.ones(num_channels) * BOS_TOKEN, 1).reshape([-1, 1]).astype('int16')
        end_token = np.tile(np.ones(num_channels) * EOS_TOKEN, 1).reshape([-1, 1]).astype('int16')
        data = np.concatenate((start_token, data, end_token), axis=1)

        # Convert label to appropriate format
        label = self.get_change_type(label)
        return data.T, label

    def get_change_type(self, source_label):
        """ Process label sequence to include BOS, EOS, and padding """
        if source_label.min() == source_label.max():
            return np.array([BOS_TOKEN, source_label[0], EOS_TOKEN, PAD_TOKEN, PAD_TOKEN])
        idx = np.where((source_label[1:] - source_label[:-1]) != 0)[0]
        idx = np.append(idx, idx[-1] + 1)
        return np.concatenate(
            ([BOS_TOKEN], source_label[idx], [EOS_TOKEN], [PAD_TOKEN] * (3 - len(idx)))
        )

    def __len__(self):
        return self.dataset.shape[0]  # Return dataset size


def load_data(area):
    """
    Load and preprocess dataset for training, validation, and testing.
    :param area: Geographic area identifier for dataset loading
    :return: Data loaders for training, validation, and test sets
    """
    selected_months = config.selected_months - 1
    dataset = np.load(rf'data/{area}.npy')[:, :11]  # Load dataset (first 11 months)

    np.random.shuffle(dataset)  # Shuffle dataset
    num_samples = dataset.shape[0]

    # Split dataset into training (80%), validation (10%), and testing (10%)
    train_set = dataset[:int(num_samples * 0.8)]
    valid_set = dataset[int(num_samples * 0.8):int(num_samples * 0.9)]
    test_set = dataset[int(num_samples * 0.9):]

    # Extract year-wise data (assuming two years of monthly data)
    train_2020, train_2021 = train_set[:, :, :12], train_set[:, :, 12:24]
    valid_2020, valid_2021 = valid_set[:, :, :12], valid_set[:, :, 12:24]
    test_2020, test_2021 = test_set[:, :, :12], test_set[:, :, 12:24]

    # Concatenate and extract the selected month for training
    train_dataset = np.concatenate((train_2020, train_2021), axis=0)[:, :, selected_months]
    valid_dataset = np.concatenate((valid_2020, valid_2021), axis=0)[:, :, selected_months]
    test_dataset = np.concatenate((test_2020, test_2021), axis=0)[:, :, selected_months]

    # Create dataset objects
    train_data = MaskDataset(train_dataset[:, :10], train_dataset[:, 10])
    valid_data = MaskDataset(valid_dataset[:, :10], valid_dataset[:, 10])
    test_data = MaskDataset(test_dataset[:, :10], test_dataset[:, 10])

    # Create data loaders
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    return train_loader, valid_loader, test_loader
