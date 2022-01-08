# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Speech to text sequence-to-sequence model."""

import numpy as np
import random
import torch
import torch.nn as nn

from models.base import ModelBase
from models.seq2seq.encoders.build import build_encoder
from models.seq2seq.decoders.build import build_decoder
from models.torch_utils import pad_list, np2tensor

class Speech2Text(ModelBase):
    """Speech to text sequence-to-sequence model."""

    def __init__(self, args, save_path=None, idx2token=None):

        super(ModelBase, self).__init__()

        # for decoder
        self.vocab = args.vocab
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for the sub tasks
        self.main_weight = args.total_weight - args.sub1_weight - args.sub2_weight

        # for CTC
        self.ctc_weight = min(args.ctc_weight, self.main_weight)

        # for backward decoder
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight

        # Feature extraction
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices

        # Encoder
        self.enc = build_encoder(args)

        special_symbols = {
            'blank': self.blank,
            'unk': self.unk,
            'eos': self.eos,
            'pad': self.pad,
        }

        # main task
        directions = []
        if self.fwd_weight > 0 or (self.bwd_weight == 0 and self.ctc_weight > 0):
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')

        for dir in directions:
            # Decoder
            dec = build_decoder(args, special_symbols,
                                self.enc.output_dim,
                                args.vocab,
                                self.ctc_weight,
                                self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight)
            setattr(self, 'dec_' + dir, dec)

    def encode(self, xs, task='all'):
        """Encode acoustic or text features.
        Args:
            xs (List): length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all/ys*/ys_sub1*/ys_sub2*
        Returns:
            eout_dict (dict):
        """
        # Frame stacking
        if self.n_stacks > 1:
            xs = [stack_frame(x, self.n_stacks, self.n_skips) for x in xs]
        # Splicing
        if self.n_splices > 1:
            xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]
        xlens = torch.IntTensor([len(x) for x in xs])
        
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0.).to(self.device)

        # encoder
        eout_dict = self.enc(xs, xlens, task.split('.')[0])
        return eout_dict

    def decode(self, xs, params):
        dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
        # Encode input features
        eout_dict = self.encode(xs, 'ys')
        eouts = eout_dict['ys']['xs']
        elens = eout_dict['ys']['xlens']
        # CTC
        if (self.fwd_weight == 0 and self.bwd_weight == 0) or (self.ctc_weight > 0 and params['recog_ctc_weight'] == 1):
            nbest_hyps_id = getattr(self, 'dec_' + dir).ctc.greedy(eouts, elens)
            return nbest_hyps_id
        elif params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
            best_hyps_id = getattr(self, 'dec_' + dir).greedy(eouts, elens)
        else:
            best_hyps_id = getattr(self, 'dec_' + dir).beam_search(eouts, elens, params)

        return best_hyps_id
