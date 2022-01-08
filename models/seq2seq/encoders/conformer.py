# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder."""

import copy
import logging
import torch.nn as nn

from models.seq2seq.encoders.conv import ConvEncoder
from models.seq2seq.encoders.conformer_block import ConformerEncoderBlock
from models.seq2seq.encoders.conformer_block_v2 import ConformerEncoderBlock_v2
from models.seq2seq.encoders.transformer import TransformerEncoder

logger = logging.getLogger(__name__)


class ConformerEncoder(TransformerEncoder):
    def __init__(self, input_dim, enc_type, n_heads, kernel_size, normalization,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
                 pe_type, layer_norm_eps, last_proj_dim,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 subsample, subsample_type, n_stacks, n_splices, frontend_conv,
                 task_specific_layer, param_init, clamp_len,
                 lookahead, chunk_size_left, chunk_size_current, chunk_size_right,
                 streaming_type):

        super(ConformerEncoder, self).__init__(
            input_dim, enc_type, n_heads,
            n_layers, n_layers_sub1, n_layers_sub2,
            d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
            pe_type, layer_norm_eps, last_proj_dim,
            dropout_in, dropout, dropout_att, dropout_layer,
            subsample, subsample_type, n_stacks, n_splices, frontend_conv,
            task_specific_layer, param_init, clamp_len,
            lookahead, chunk_size_left, chunk_size_current, chunk_size_right,
            streaming_type)

        causal = self.unidir or (self.streaming_type == 'mask')
        if 'conformer_v2' in enc_type:
            conformer_block = ConformerEncoderBlock_v2
        else:
            assert pe_type in ['relative', 'relative_xl']
            conformer_block = ConformerEncoderBlock

        self.layers = nn.ModuleList([copy.deepcopy(conformer_block(
            d_model, d_ff, n_heads, kernel_size,
            dropout, dropout_att, dropout_layer * (lth + 1) / n_layers,
            layer_norm_eps, ffn_activation, param_init,
            pe_type, clamp_len, ffn_bottleneck_dim, causal, normalization))
            for lth in range(n_layers)])
