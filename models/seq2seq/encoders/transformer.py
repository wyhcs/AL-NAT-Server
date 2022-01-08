# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder."""

import copy
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from models.modules.positional_embedding import (
    PositionalEncoding,
    XLPositionalEmbedding
)
from models.seq2seq.encoders.conv import ConvEncoder
from models.seq2seq.encoders.encoder_base import EncoderBase
from models.seq2seq.encoders.subsampling import (
    AddSubsampler,
    ConcatSubsampler,
    Conv1dSubsampler,
    DropSubsampler,
    MaxPoolSubsampler,
    MeanPoolSubsampler
)
from models.seq2seq.encoders.transformer_block import TransformerEncoderBlock
from models.seq2seq.encoders.utils import chunkwise
from models.torch_utils import (
    make_pad_mask,
    tensor2np
)

class TransformerEncoder(EncoderBase):
    def __init__(self, input_dim, enc_type, n_heads,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
                 pe_type, layer_norm_eps, last_proj_dim,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 subsample, subsample_type, n_stacks, n_splices, frontend_conv,
                 task_specific_layer, param_init, clamp_len,
                 lookahead, chunk_size_left, chunk_size_current, chunk_size_right, streaming_type):

        super(TransformerEncoder, self).__init__()

        # parse subsample
        self.subsample_factors = [1] * n_layers
        for lth, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            self.subsample_factors[lth] = s
        # parse lookahead
        lookaheads = [0] * n_layers
        for lth, s in enumerate(list(map(int, lookahead.split('_')[:n_layers]))):
            lookaheads[lth] = s

        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise Warning('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' %
                          (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise Warning('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' %
                          (n_layers_sub1, n_layers_sub2))

        self.enc_type = enc_type
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.scale = math.sqrt(d_model)

        # for compatibility
        chunk_size_left = str(chunk_size_left)
        chunk_size_current = str(chunk_size_current)
        chunk_size_right = str(chunk_size_right)

        # for streaming encoder
        self.unidir = 'uni' in enc_type
        self.lookaheads = lookaheads
        if sum(lookaheads) > 0:
            assert self.unidir
        self.N_l = int(chunk_size_left.split('_')[-1]) // n_stacks
        self.N_c = int(chunk_size_current.split('_')[-1]) // n_stacks
        self.N_r = int(chunk_size_right.split('_')[-1]) // n_stacks
        self.lc_bidir = self.N_c > 0 and enc_type != 'conv' and 'uni' not in enc_type
        self.cnn_lookahead = self.unidir or enc_type == 'conv'
        self.streaming_type = streaming_type if self.lc_bidir else ''
        self.causal = self.unidir or self.streaming_type == 'mask'
      
        if self.unidir:
            assert self.N_l == self.N_c == self.N_r == 0
        if self.streaming_type == 'mask':
            assert self.N_r == 0
            assert self.N_l % self.N_c == 0
            # NOTE: this is important to cache CNN output at each chunk
        if self.lc_bidir:
            assert n_layers_sub1 == 0
            assert n_layers_sub2 == 0
            assert not self.unidir

        # for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # for bridge layers
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # Setting for frontend CNNs
        self.conv = frontend_conv
        if self.conv is not None:
            self._odim = self.conv.output_dim
        else:
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)

        # calculate subsampling factor
        self._factor = 1
        self.conv_factor = self.conv.subsampling_factor if self.conv is not None else 1
        self._factor *= self.conv_factor
        self.subsample_layers = None
        if np.prod(self.subsample_factors) > 1:
            self._factor *= np.prod(self.subsample_factors)
            if subsample_type == 'max_pool':
                self.subsample_layers = nn.ModuleList([MaxPoolSubsampler(factor)
                                                       for factor in self.subsample_factors])
            elif subsample_type == 'mean_pool':
                self.subsample_layers = nn.ModuleList([MeanPoolSubsampler(factor)
                                                       for factor in self.subsample_factors])
            elif subsample_type == 'concat':
                self.subsample_layers = nn.ModuleList([ConcatSubsampler(factor, self._odim)
                                                       for factor in self.subsample_factors])
            elif subsample_type == 'drop':
                self.subsample_layers = nn.ModuleList([DropSubsampler(factor)
                                                       for factor in self.subsample_factors])
            elif subsample_type == 'conv1d':
                assert not self.causal
                self.subsample_layers = nn.ModuleList([Conv1dSubsampler(factor, self._odim)
                                                       for factor in self.subsample_factors])
            elif subsample_type == 'add':
                self.subsample_layers = nn.ModuleList([AddSubsampler(factor)
                                                       for factor in self.subsample_factors])
            else:
                raise NotImplementedError(subsample_type)

        assert self.N_l % self._factor == 0
        assert self.N_c % self._factor == 0
        assert self.N_r % self._factor == 0

        self.pos_enc, self.pos_emb = None, None
        self.u_bias, self.v_bias = None, None
        if pe_type in ['relative', 'relative_xl']:
            self.pos_emb = XLPositionalEmbedding(d_model, dropout)
            if pe_type == 'relative_xl':
                self.u_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
                self.v_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
                # NOTE: u_bias and v_bias are global parameters shared in the whole model
        else:
            self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type, param_init)

        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderBlock(
            d_model, d_ff, n_heads,
            dropout, dropout_att, dropout_layer * (lth + 1) / n_layers,
            layer_norm_eps, ffn_activation, param_init,
            pe_type, clamp_len, ffn_bottleneck_dim))
            for lth in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._odim = d_model

        self.cache = [None] * self.n_layers
        

    def forward(self, xs, xlens, task, streaming=False,
                lookback=False, lookahead=False):
      
        eouts = {'ys': {'xs': None, 'xlens': None}}

        bs, xmax = xs.size()[:2]
        n_chunks = 0
        unidir = self.unidir
        lc_bidir = self.lc_bidir
        N_l, N_c, N_r = self.N_l, self.N_c, self.N_r

        if streaming and self.streaming_type == 'mask':
            assert xmax <= N_c
        elif streaming and self.streaming_type == 'reshape':
            assert xmax <= (N_l + N_c + N_r)


        # Path through CNN blocks
        xs, xlens = self.conv(xs, xlens,
                              lookback=False if lc_bidir else lookback,
                              lookahead=False if lc_bidir else lookahead)

        # NOTE: CNN lookahead surpassing a chunk is not allowed in chunkwise processing
        N_l = max(0, N_l // self.conv_factor)
        N_c = N_c // self.conv_factor
        N_r = N_r // self.conv_factor
        emax = xs.size(1)

        if self.enc_type == 'conv':
            eouts['ys']['xs'] = xs
            eouts['ys']['xlens'] = xlens
            return eouts

        n_cache = self.cache[0]['input_san'].size(1) if streaming and self.cache[0] is not None else 0

        # positional encoding
        if 'relative' in self.pe_type:
            xs, rel_pos_embs = self.pos_emb(xs, scale=True, n_cache=n_cache)
        else:
            xs = self.pos_enc(xs, scale=True, offset=self.offset)
            rel_pos_embs = None

        new_cache = [None] * self.n_layers
       
        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[0])
        for lth, layer in enumerate(self.layers):
            xs, cache = layer(xs, xx_mask, cache=self.cache[lth],
                              pos_embs=rel_pos_embs, rel_bias=(self.u_bias, self.v_bias))
            new_cache[lth] = cache

            # Pick up outputs in the sub task before the projection layer
            if lth == self.n_layers_sub1 - 1:
                xs_sub1 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub1')
                xlens_sub1 = xlens.clone()
                if task == 'ys_sub1':
                    eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                    return eouts
            if lth == self.n_layers_sub2 - 1:
                xs_sub2 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub2')
                xlens_sub2 = xlens.clone()
                if task == 'ys_sub2':
                    eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                    return eouts

          
            if lth < len(self.layers) - 1:
                if self.subsample_factors[lth] > 1:
                    xs, xlens = self.subsample_layers[lth](xs, xlens)
                if streaming:
                    # This is necessary at every layer during streaming inference because of different cache sizes
                    n_cache = self.cache[lth + 1]['input_san'].size(
                        1) if streaming and self.cache[lth + 1] is not None else 0
                    if 'relative' in self.pe_type:
                        xs, rel_pos_embs = self.pos_emb(xs, n_cache=n_cache)
                    xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])
                else:
                    if self.subsample_factors[lth] > 1:
                        if 'relative' in self.pe_type:
                            xs, rel_pos_embs = self.pos_emb(xs)
                        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])
                    elif self.lookaheads[lth] != self.lookaheads[lth + 1]:
                        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])

        xs = self.norm_out(xs)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        return eouts

def make_san_mask(xs, xlens, unidirectional=False, lookahead=0):
    """Mask self-attention mask.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        unidirectional (bool): pad future context
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_pad_mask(xlens.to(xs.device))
    xx_mask = xx_mask.unsqueeze(1).repeat([1, xlens.max(), 1])  # `[B, emax (query), emax (key)]`
    if unidirectional:
        xx_mask = causal(xx_mask, lookahead)
    return xx_mask


def causal(xx_mask, lookahead):
    """Causal masking.

    Args:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    causal_mask = xx_mask.new_ones(xx_mask.size(1), xx_mask.size(1), dtype=xx_mask.dtype)
    causal_mask = torch.tril(causal_mask, diagonal=lookahead, out=causal_mask).unsqueeze(0)
    xx_mask = xx_mask & causal_mask  # `[B, L (query), L (key)]`
    return xx_mask


def make_chunkwise_san_mask(xs, xlens, N_l, N_c, n_chunks):
    """Mask self-attention mask for chunkwise processing.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        n_chunks (int): number of chunks
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_san_mask(xs, xlens)
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * N_c
        xx_mask[:, offset:offset + N_c, :max(0, offset - N_l)] = 0
        xx_mask[:, offset:offset + N_c, offset + N_c:] = 0
    return xx_mask
