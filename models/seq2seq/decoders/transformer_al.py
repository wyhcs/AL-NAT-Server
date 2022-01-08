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
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from models.modules.positional_embedding import PositionalEncoding
from models.modules.transformer import TransformerDecoderBlock
from models.seq2seq.decoders.ctc import CTC
from models.seq2seq.decoders.decoder_base import DecoderBase
from models.torch_utils import (
    append_sos_eos,
    compute_accuracy,
    calculate_cer_ctc,
    calculate_cer,
    calculate_wer,
    make_pad_mask,
    tensor2np,
    tensor2scalar
)

from models.seq2seq.lmdecode.decoder import build_ctcdecoder
import kenlm

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerDecoder(DecoderBase):
    def __init__(self, special_symbols,
                 enc_n_units, attn_type, n_heads, n_layers,
                 d_model, d_ff, ffn_bottleneck_dim,
                 pe_type, layer_norm_eps, ffn_activation,
                 vocab, tie_embedding,
                 dropout, dropout_emb, dropout_att, dropout_layer, dropout_head,
                 lsm_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, backward,
                 global_weight, mtl_per_batch, param_init,
                 mma_chunk_size, mma_n_heads_mono, mma_n_heads_chunk,
                 mma_init_r, mma_eps, mma_std,
                 mma_no_denominator, mma_1dconv,
                 mma_quantity_loss_weight, mma_headdiv_loss_weight,
                 latency_metric, latency_loss_weight,
                 mma_first_layer, share_chunkwise_attention,
                 external_lm, lm_fusion,
                 do_bac, do_normal, gamma, alpha, recog_alpha, recog_beta, args):

        super(TransformerDecoder, self).__init__()
        
        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight

        # for cache
        self.embed_cache = None

        # for MMA
        self.attn_type = attn_type
        self.quantity_loss_weight = mma_quantity_loss_weight
        self._quantity_loss_weight = mma_quantity_loss_weight  # for curriculum
        self.mma_first_layer = max(1, mma_first_layer)
        self.headdiv_loss_weight = mma_headdiv_loss_weight

        if ctc_weight > 0:
            self.ctc = CTC(blank=self.blank, enc_n_units=enc_n_units, fc_list=ctc_fc_list,  vocab=vocab,)

        self.corpus = args.corpus
        if args.corpus == 'aishell':
            self.labels_aishell = [""]
            with open(args.dict, 'r', encoding='UTF-8') as f:
                for x in f:
                    self.labels_aishell.append(x.split()[0].replace('<space>', ' '))


        if args.unit == 'char' and int(args.recog_beam_width )>1:
            self.TEST_KENLM_MODEL = kenlm.Model(args.kenlm_path)
            self.labels_char = [""]
            with open(args.dict, 'r', encoding='UTF-8') as f:
                for x in f:
                    self.labels_char.append(x.split()[0].replace('<space>', ' '))
            self.decoder = build_ctcdecoder(self.labels_char, self.TEST_KENLM_MODEL, alpha=args.recog_alpha, beta=args.recog_beta, is_bpe=False, ctc_token_idx=0)
        elif args.unit == 'wp' and int(args.recog_beam_width )>1:
            self.TEST_KENLM_MODEL = kenlm.Model(args.kenlm_path)
            self.labels_bpe = [""]
            with open(args.dict, 'r', encoding='UTF-8') as f:
                for x in f:
                    self.labels_bpe.append(x.split()[0])
            self.decoder = build_ctcdecoder(self.labels_bpe, self.TEST_KENLM_MODEL, alpha=args.recog_alpha, beta=args.recog_beta, is_bpe=True, ctc_token_idx=0, bpe_path=args.wp_model)


        if self.att_weight > 0:

            if args.transformer_enc_d_model != args.transformer_dec_d_model:
                self.bridge = nn.Linear(args.transformer_enc_d_model, args.transformer_dec_d_model)
            else:
                self.bridge = None

            # token embedding
            self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type, param_init)
            # decoder
            self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
                d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
                layer_norm_eps, ffn_activation, param_init,
                src_tgt_attention=False if lth < mma_first_layer - 1 else True,
                mma_chunk_size=mma_chunk_size,
                mma_n_heads_mono=mma_n_heads_mono,
                mma_n_heads_chunk=mma_n_heads_chunk,
                mma_init_r=mma_init_r,
                mma_eps=mma_eps,
                mma_std=mma_std,
                mma_no_denominator=mma_no_denominator,
                mma_1dconv=mma_1dconv,
                dropout_head=dropout_head,
                lm_fusion=lm_fusion,
                ffn_bottleneck_dim=ffn_bottleneck_dim,
                share_chunkwise_attention=share_chunkwise_attention)) for lth in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                self.output.weight = self.embed.weight


    def greedy(self, eouts, elens):
        ys_in = self.ctc.greedy(eouts, elens)

        [ ys_in[0].insert(i, 0) for i in range(len(ys_in[0])-1, 0, -1) if ys_in[0][i] == ys_in[0][i-1]] 
        ys_in = torch.LongTensor(ys_in).to(self.device)
        # return ys_in.cpu().detach().numpy().tolist(), None

        for b in range(eouts.size(0)):
            [ ys_in[b].insert(i, 0) for i in range(len(ys_in[b])-1, 0, -1) if ys_in[b][i] == ys_in[b][i-1]] 
        try:
            out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout
        except:
            return ''

        if self.bridge is not None:
            eouts = self.bridge(eouts)
        for lth, layer in enumerate(self.layers):
            out = layer(out, None, eouts, None, mode='parallel')
        logits = self.output(self.norm_out(out))
        logits = F.log_softmax(logits, dim=2)
        ys_hat = logits.argmax(dim=-1).cpu().detach().numpy().tolist()
        ys_hat = [x[0] for x in groupby(ys_hat[0])]
        ys_hat = [x for x in filter(lambda x: x != 0, ys_hat)]

        if self.corpus == 'aishell':
            ys_hat = ''.join([self.labels_aishell[x] for x in ys_hat])

        return ys_hat

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


    def beam_search(self, eouts, elens, params):
        
        beam_width = params.get('recog_beam_width')
        logits = self.ctc.log_softmax(eouts).detach().cpu().numpy()[0]
        ys_in = [self.decoder.decode(logits, beam_width=beam_width, is_train=True)]
        [ ys_in[0].insert(i, 0) for i in range(len(ys_in[0])-1, 0, -1) if ys_in[0][i] == ys_in[0][i-1]] 
        ys_in = torch.LongTensor(ys_in).to(self.device)

        try:
            out = self.pos_enc(self.embed_token_id(ys_in), scale=True).to(self.device)  # scaled + dropout
        except:
            return ''

        if self.bridge is not None:
            eouts = self.bridge(eouts)
        for lth, layer in enumerate(self.layers):
            out = layer(out, None, eouts, None, mode='parallel')

        # logits = self.output(self.norm_out(out))
        # logits = F.log_softmax(logits, dim=2)
        # ys_hat = logits.argmax(dim=-1).cpu().numpy().tolist()
        # ys_hat = [x[0] for x in groupby(ys_hat[0])]
        # text_list = [x for x in filter(lambda x: x != 0, ys_hat)]
        # text_list = ''.join([self.labels_bpe[hyp_id] for hyp_id in text_list]).replace('▁',' ').strip()

        logits = self.output(self.norm_out(out))
        logits = F.log_softmax(logits, dim=2).detach().cpu().numpy()[0]
        text_list = self.decoder.decode(logits, beam_width=beam_width, is_train=False)

        return text_list
