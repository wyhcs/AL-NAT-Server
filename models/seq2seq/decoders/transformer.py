# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

import copy
from distutils.util import strtobool
from distutils.version import LooseVersion
import logging
import math
import numpy as np
import random
from itertools import groupby
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import codecs
import sys

sys.path.append('/data/wyh/transformer/server/')

from models.modules.positional_embedding import PositionalEncoding
from models.modules.transformer import TransformerDecoderBlock


random.seed(1)


class TransformerDecoder(nn.Module):
    def __init__(self):

        super(TransformerDecoder, self).__init__()
        
        self.pad = 1
        self.enc_n_units = 512
        self.attn_type = 'scaled_dot'
        self.n_heads = 8
        self.n_layers = 6
        self.d_model = 512
        self.d_ff = 2048
        self.ffn_bottleneck_dim = 0
        self.pe_type = '1dconv3L'
        self.layer_norm_eps = 1e-12
        self.ffn_activation = 'swish'
        self.vocab = 9858
        self.tie_embedding = False
        self.dropout = 0.1
        self.dropout_emb = 0.1
        self.dropout_att = 0.0
        self.dropout_layer = 0.0
        self.dropout_head = 0.0
        self.lsm_prob = 0.1
        self.ctc_weight = 0.8
        self.ctc_lsm_prob = 0.1
        self.ctc_fc_list = 512
        self.backward = False
        self.global_weight = 1.0
        self.mtl_per_batch = False
        self.param_init = 'xavier_uniform'
        self.mma_chunk_size = 1
        self.mma_n_heads_mono = 1 
        self.mma_n_heads_chunk = 1
        self.mma_init_r = -4
        self.mma_eps = 1e-06
        self.mma_std = 1.0
        self.mma_no_denominator = False
        self.mma_1dconv = False
        self.mma_quantity_loss_weight = 0.0
        self.mma_headdiv_loss_weight = 0.0
        self.latency_metric = ''
        self.latency_loss_weight = 0.0
        self.mma_first_layer = 1
        self.share_chunkwise_attention = False
        self.external_lm = None
        self.lm_fusion = ''

        # for cache
        self.embed_cache = None

        # for MMA
        self.quantity_loss_weight = 0.0
        self._quantity_loss_weight = 0.0  # for curriculum
        self.mma_first_layer = 1
        self.headdiv_loss_weight = 0.0


        # self.bridge = nn.Linear(args.transformer_enc_d_model, args.transformer_dec_d_model)

        # token embedding
        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.pos_enc = PositionalEncoding(self.d_model, self.dropout_emb, self.pe_type, self.param_init)
        # decoder
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
            self.d_model, self.d_ff, self.attn_type, self.n_heads, self.dropout, self.dropout_att, self.dropout_layer,
            self.layer_norm_eps, self.ffn_activation, self.param_init,
            src_tgt_attention=False if lth < self.mma_first_layer - 1 else True,
            mma_chunk_size=self.mma_chunk_size,
            mma_n_heads_mono=self.mma_n_heads_mono,
            mma_n_heads_chunk=self.mma_n_heads_chunk,
            mma_init_r=self.mma_init_r,
            mma_eps=self.mma_eps,
            mma_std=self.mma_std,
            mma_no_denominator=self.mma_no_denominator,
            mma_1dconv=self.mma_1dconv,
            dropout_head=self.dropout_head,
            lm_fusion=self.lm_fusion,
            ffn_bottleneck_dim=self.ffn_bottleneck_dim,
            share_chunkwise_attention=self.share_chunkwise_attention)) for lth in range(self.n_layers)])
        self.norm_out = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        self.output = nn.Linear(self.d_model, self.vocab)
        # if tie_embedding:
        #     self.output.weight = self.embed.weight

    def embed_token_id(self, indices):
        """Embed token IDs.
        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.embed(indices)
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def forward(self, ys_in):
        xlens = [len(x) for x in ys_in]
        th_ilen = torch.from_numpy(np.array(xlens))

        ys_in = pad_sequence([torch.LongTensor(f) for f in ys_in], batch_first=True)

        out = self.pos_enc(self.embed_token_id(ys_in), scale=True)

        for lth, layer in enumerate(self.layers):
            out = layer(out, None, None, None, mode='parallel')
        return out



model = TransformerDecoder()

print(model)

# ys_in = [[5,6,7], [8,9,10]]

# print(model(ys_in).shape)






    # def greedy(self, eouts, elens):
    #     ys_in = self.ctc.greedy(eouts, elens)

    #     [ ys_in[0].insert(i, 0) for i in range(len(ys_in[0])-1, 0, -1) if ys_in[0][i] == ys_in[0][i-1]] 
    #     ys_in = torch.LongTensor(ys_in).to(self.device)
    #     # return ys_in.cpu().numpy().tolist(), None

    #     for b in range(eouts.size(0)):
    #         [ ys_in[b].insert(i, 0) for i in range(len(ys_in[b])-1, 0, -1) if ys_in[b][i] == ys_in[b][i-1]] 
    #     try:
    #         out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout
    #     except:
    #         ys_in = torch.LongTensor([np.random.randint(4, self.vocab, int(elens//3))]).to(self.device)
    #         out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout

    #     if self.bridge is not None:
    #         eouts = self.bridge(eouts)
    #     for lth, layer in enumerate(self.layers):
    #         out = layer(out, None, eouts, None, mode='parallel')
    #     logits = self.output(self.norm_out(out))
    #     logits = F.log_softmax(logits, dim=2)
    #     ys_hat = logits.argmax(dim=-1).cpu().numpy().tolist()
    #     ys_hat = [x[0] for x in groupby(ys_hat[0])]
    #     ys_hat = [x for x in filter(lambda x: x != 0, ys_hat)]

    #     return ys_hat




    # def beam_search(self, eouts, elens, params):
        
    #     beam_width = params.get('recog_beam_width')
    #     # ys_in = self.ctc.greedy(eouts, elens)

    #     logits = self.ctc.log_softmax(eouts).detach().cpu().numpy()[0]
    #     ys_in = [self.decoder.decode(logits, beam_width=beam_width, is_train=True)]
    #     [ ys_in[0].insert(i, 0) for i in range(len(ys_in[0])-1, 0, -1) if ys_in[0][i] == ys_in[0][i-1]] 
    #     ys_in = torch.LongTensor(ys_in).to(self.device)

    #     try:
    #         out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout
    #     except:
    #         ys_in = torch.LongTensor([np.random.randint(4, self.vocab, int(elens//3))]).to(self.device)
    #         out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout

    #     if self.bridge is not None:
    #         eouts = self.bridge(eouts)
    #     for lth, layer in enumerate(self.layers):
    #         out = layer(out, None, eouts, None, mode='parallel')

    #     # logits = self.output(self.norm_out(out))
    #     # logits = F.log_softmax(logits, dim=2)
    #     # ys_hat = logits.argmax(dim=-1).cpu().numpy().tolist()
    #     # ys_hat = [x[0] for x in groupby(ys_hat[0])]
    #     # text_list = [x for x in filter(lambda x: x != 0, ys_hat)]
    #     # text_list = ''.join([self.libri_labels_bpe[hyp_id] for hyp_id in text_list]).replace('▁',' ').strip()

    #     logits = self.output(self.norm_out(out))
    #     logits = F.log_softmax(logits, dim=2).detach().cpu().numpy()[0]
    #     text_list = self.decoder.decode(logits, beam_width=beam_width, is_train=False)

    #     return text_list


