# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

import logging
import numpy as np
import os
import shutil

from models.base import ModelBase

class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):
        super(ModelBase, self).__init__()

    def reset_session(self):
        self._new_session = True

    def trigger_scheduled_sampling(self):
        self._ss_prob = getattr(self, 'ss_prob', 0)

    def trigger_quantity_loss(self):
        self._quantity_loss_weight = getattr(self, 'quantity_loss_weight', 0)

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def embed_token_id(self, indices):
        raise NotImplementedError

    def cache_embedding(self, device):
        raise NotImplementedError


