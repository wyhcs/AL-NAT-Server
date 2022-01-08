# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CTC decoder."""

from collections import OrderedDict
from distutils.version import LooseVersion
from itertools import groupby
import torch
import torch.nn as nn

from models.seq2seq.decoders.decoder_base import DecoderBase

class CTC(DecoderBase):
    def __init__(self, blank, enc_n_units, fc_list, vocab):
        super(CTC, self).__init__()

        self.blank = blank
        # Fully-connected layers before the softmax
        if fc_list is not None and len(fc_list) > 0:
            _fc_list = [int(fc) for fc in fc_list.split('_')]
            fc_layers = OrderedDict()
            for i in range(len(_fc_list)):
                input_dim = enc_n_units if i == 0 else _fc_list[i - 1]
                fc_layers['fc' + str(i)] = nn.Linear(input_dim, _fc_list[i])
            fc_layers['fc' + str(len(_fc_list))] = nn.Linear(_fc_list[-1], vocab)
            self.output = nn.Sequential(fc_layers)
        else:
            self.output = nn.Linear(enc_n_units, vocab)

    def log_softmax(self, eouts):
        return torch.log_softmax(self.output(eouts), dim=-1)

    def greedy(self, eouts, elens):
        """Greedy decoding.
        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (np.ndarray): `[B]`
        Returns:
            hyps (np.ndarray): Best path hypothesis. `[B, L]`
        """
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        best_paths = log_probs.argmax(-1)  # `[B, L]`

        hyps = []
        for b in range(eouts.size(0)):
            indices = [best_paths[b, t].item() for t in range(elens[b])]
            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]
            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            hyps.append(best_hyp)
        return hyps
