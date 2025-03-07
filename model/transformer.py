#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Config import config

# Initialize the configuration
config = config()
# Get the device (CPU or GPU) from the configuration
DEVICE = config.device

class Embeddings(nn.Module):
    """
    This class is used to convert input tokens into embeddings.
    It provides different embedding methods for the encoder and decoder.
    """
    def __init__(self, d_model, vocab, func='encoder'):
        """
        Initialize the Embeddings class.

        Args:
            d_model (int): The dimensionality of the embedding vectors.
            vocab (int): The size of the vocabulary.
            func (str, optional): Specify whether it is used for the encoder or decoder. Defaults to 'encoder'.
        """
        super(Embeddings, self).__init__()
        # Linear layer for the encoder
        self.lut_encoder = nn.Linear(vocab, d_model)
        # Embedding layer for the decoder
        self.lut_decoder = nn.Embedding(vocab, d_model)
        # Dimensionality of the embedding vectors
        self.d_model = d_model
        # Function type (encoder or decoder)
        self.func = func

    def forward(self, x):
        """
        Perform the forward pass of the embedding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor.
        """
        x = x.to(DEVICE)
        if self.func != 'encoder':
            x = x.type(torch.long)
            return self.lut_decoder(x) * math.sqrt(self.d_model)
        else:
            return self.lut_encoder(x) * math.sqrt(self.d_model)


class PositionalEncoding_Order(nn.Module):
    """
    This class adds positional encoding to the input embeddings to capture the position information.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        Initialize the PositionalEncoding_Order class.

        Args:
            d_model (int): The dimensionality of the embedding vectors.
            dropout (float): The dropout rate.
            max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.
        """
        super(PositionalEncoding_Order, self).__init__()
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Initialize the positional encoding matrix
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # Create a tensor representing positions
        position = torch.arange(0.0, max_len, device=DEVICE)
        position.unsqueeze_(1)
        # Calculate the division term for positional encoding
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # Calculate sine and cosine values for positional encoding
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        pe.unsqueeze_(0)
        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Perform the forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    """
    Clone a given module N times. The cloned modules do not share parameters.

    Args:
        module (nn.Module): The module to be cloned.
        N (int): The number of clones.

    Returns:
        nn.ModuleList: A list containing N cloned modules.
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(N)
    ])


def attention(query, key, value, mask=None, dropout=None):
    """
    Calculate the Scaled Dot - Product Attention as described in equation (4).

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        mask (torch.Tensor, optional): Mask tensor. Defaults to None.
        dropout (nn.Dropout, optional): Dropout layer. Defaults to None.

    Returns:
        tuple: A tuple containing the output tensor and the attention weights.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Implement the Multi - Head Attention mechanism as described in the encoder (2).
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        Initialize the MultiHeadedAttention class.

        Args:
            h (int): The number of attention heads.
            d_model (int): The dimensionality of the input vectors.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(MultiHeadedAttention, self).__init__()
        """
        `h`: The number of attention heads.
        `d_model`: The dimensionality of the word vectors.
        """
        assert d_model % h == 0
        # Dimensionality of each attention head
        self.d_k = d_model // h
        # Number of attention heads
        self.h = h
        # Four linear layers for query, key, value, and output
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # Attention weights
        self.attn = None
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Perform the forward pass of the multi - head attention layer.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        # Number of batches
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """
    Implement the layer normalization operation.
    """
    def __init__(self, features, eps=1e-6):
        """
        Initialize the LayerNorm class.

        Args:
            features (int): The number of features in the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__()
        # Scaling parameter
        self.a_2 = nn.Parameter(torch.ones(features))
        # Shifting parameter
        self.b_2 = nn.Parameter(torch.zeros(features))
        # Epsilon for numerical stability
        self.eps = eps

    def forward(self, x):
        """
        Perform the forward pass of the layer normalization layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2


class SublayerConnection(nn.Module):
    """
    Connect the Multi - Head Attention and Feed Forward layers through layer normalization and residual connection.
    """
    def __init__(self, size, dropout):
        """
        Initialize the SublayerConnection class.

        Args:
            size (int): The dimensionality of the input vectors.
            dropout (float): The dropout rate.
        """
        super(SublayerConnection, self).__init__()
        # Layer normalization layer
        self.norm = LayerNorm(size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Perform the forward pass of the sub - layer connection.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (function): The sub - layer function.

        Returns:
            torch.Tensor: Output tensor after the sub - layer operation and residual connection.
        """
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        return x + x_


class PositionwiseFeedForward(nn.Module):
    """
    Implement the position - wise feed - forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize the PositionwiseFeedForward class.

        Args:
            d_model (int): The dimensionality of the input vectors.
            d_ff (int): The dimensionality of the hidden layer in the feed - forward network.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(PositionwiseFeedForward, self).__init__()
        # First linear layer
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second linear layer
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the position - wise feed - forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Define a single layer of the encoder.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initialize the EncoderLayer class.

        Args:
            size (int): The dimensionality of the input vectors.
            self_attn (nn.Module): The self - attention layer.
            feed_forward (nn.Module): The feed - forward network.
            dropout (float): The dropout rate.
        """
        super(EncoderLayer, self).__init__()
        # Self - attention layer
        self.self_attn = self_attn
        # Feed - forward network
        self.feed_forward = feed_forward
        # Two sub - layer connections
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # Dimensionality of the input vectors
        self.size = size

    def forward(self, x, mask):
        """
        Perform the forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mask = mask.to(DEVICE)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Define the encoder module.
    """
    def __init__(self, layer, N):
        """
        Initialize the Encoder class.

        Args:
            layer (nn.Module): The basic encoder layer.
            N (int): The number of encoder layers.
        """
        super(Encoder, self).__init__()
        """
        layer = EncoderLayer
        """
        # Multiple encoder layers
        self.layers = clones(layer, N)
        # Layer normalization layer
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Perform the forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor after passing through all encoder layers and layer normalization.
        """
        """
        Loop through the basic encoder layer N times.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Define the decoder module.
    """
    def __init__(self, layer, N):
        """
        Initialize the Decoder class.

        Args:
            layer (nn.Module): The basic decoder layer.
            N (int): The number of decoder layers.
        """
        super(Decoder, self).__init__()
        # Multiple decoder layers
        self.layers = clones(layer, N)
        # Layer normalization layer
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Perform the forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor after passing through all decoder layers and layer normalization.
        """
        """
        Loop through the basic decoder layer N times.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Define a single layer of the decoder.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Initialize the DecoderLayer class.

        Args:
            size (int): The dimensionality of the input vectors.
            self_attn (nn.Module): The self - attention layer.
            src_attn (nn.Module): The source - attention layer.
            feed_forward (nn.Module): The feed - forward network.
            dropout (float): The dropout rate.
        """
        super(DecoderLayer, self).__init__()
        # Dimensionality of the input vectors
        self.size = size
        # Self - attention layer
        self.self_attn = self_attn
        # Source - attention layer
        self.src_attn = src_attn
        # Feed - forward network
        self.feed_forward = feed_forward
        # Three sub - layer connections
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Perform the forward pass of the decoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Generator(nn.Module):
    """
    Map the decoder output to a probability distribution over the vocabulary through a linear transformation and softmax function.
    """
    def __init__(self, d_model, vocab):
        """
        Initialize the Generator class.

        Args:
            d_model (int): The dimensionality of the input vectors.
            vocab (int): The size of the vocabulary.
        """
        super(Generator, self).__init__()
        # Linear projection layer
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Perform the forward pass of the generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log - softmax output representing the probability distribution over vocabulary.
        """
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    Implement the Transformer model, which consists of an encoder, a decoder,
    source and target embedding layers, and a generator.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Initialize the Transformer class.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            src_embed (nn.Module): The source embedding layer.
            tgt_embed (nn.Module): The target embedding layer.
            generator (nn.Module): The generator module.
        """
        super(Transformer, self).__init__()
        # Encoder module
        self.encoder = encoder
        # Decoder module
        self.decoder = decoder
        # Source embedding layer
        self.src_embed = src_embed
        # Target embedding layer
        self.tgt_embed = tgt_embed
        # Generator module
        self.generator = generator

    def encode(self, src, src_mask):
        """
        Encode the source input using the encoder and source embedding layer.

        Args:
            src (torch.Tensor): Source input tensor.
            src_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Encoded output tensor.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decode the target input using the decoder, target embedding layer,
        and the output from the encoder.

        Args:
            memory (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask tensor.
            tgt (torch.Tensor): Target input tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Perform the forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor.
            tgt (torch.Tensor): Target input tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor after encoding and decoding.
        """
        encoder = self.encode(src, src_mask)
        return self.decode(encoder, src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Create a Transformer model with the specified hyperparameters.

    Args:
        src_vocab (int): The size of the source vocabulary.
        tgt_vocab (int): The size of the target vocabulary.
        N (int, optional): The number of encoder and decoder layers. Defaults to 6.
        d_model (int, optional): The dimensionality of the embedding vectors. Defaults to 512.
        d_ff (int, optional): The dimensionality of the hidden layer in the feed - forward network. Defaults to 2048.
        h (int, optional): The number of attention heads. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.

    Returns:
        nn.Module: The created Transformer model.
    """
    c = copy.deepcopy
    # Multi - Head Attention layer
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # Position - wise Feed - Forward network
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # Positional Encoding layer
    position = PositionalEncoding_Order(d_model, dropout).to(DEVICE)

    model = Transformer(
        # Encoder module
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        # Decoder module
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        # Source embedding layer and positional encoding
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        # Target embedding layer and positional encoding
        nn.Sequential(Embeddings(d_model, tgt_vocab, func='decoder').to(DEVICE), c(position)),
        # Generator module
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # Initialize the model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)

