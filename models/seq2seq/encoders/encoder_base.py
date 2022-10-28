# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for encoders."""

import os
import shutil
import torch

from models.base import ModelBase

class EncoderBase(ModelBase):
    """Base class for encoders."""

    def __init__(self):
        super(ModelBase, self).__init__()

    @property
    def output_dim(self):
        return self._odim

    @property
    def output_dim_sub1(self):
        return getattr(self, '_odim_sub1', self._odim)

    @property
    def output_dim_sub2(self):
        return getattr(self, '_odim_sub2', self._odim)

    @property
    def subsampling_factor(self):
        return self._factor

    @property
    def subsampling_factor_sub1(self):
        return self._factor_sub1

    @property
    def subsampling_factor_sub2(self):
        return self._factor_sub2

    def forward(self, xs, xlens, task):
        raise NotImplementedError

    def reset_cache(self):
        raise NotImplementedError

    def turn_on_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = True
                    logging.debug('Turn ON ceil_mode in %s.' % name)
                else:
                    self.turn_on_ceil_mode(module)

    def turn_off_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = False
                    logging.debug('Turn OFF ceil_mode in %s.' % name)
                else:
                    self.turn_off_ceil_mode(module)
