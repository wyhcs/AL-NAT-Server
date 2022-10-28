#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Token masking module for Masked LM."""

import numpy


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples)

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)
