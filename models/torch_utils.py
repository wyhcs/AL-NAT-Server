# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions."""

import copy
import numpy as np
import torch
import editdistance
from itertools import groupby

def repeat(module, n_layers):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (torch.Tensor):
    Returns:
        np.ndarray

    """
    if x is None:
        return x
    return x.cpu().detach().numpy()


def tensor2scalar(x):
    """Convert torch.Tensor to a scalar value.

    Args:
        x (torch.Tensor):
    Returns:
        scaler

    """
    if isinstance(x, float):
        return x
    return x.cpu().detach().item()


def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): A tensor of any sizes
    Returns:
        tensor (torch.Tensor):

    """
    tensor = torch.from_numpy(array).to(device)
    return tensor


def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def make_pad_mask(seq_lens):
    """Make mask for padding.

    Args:
        seq_lens (IntTensor): `[B]`
    Returns:
        mask (IntTensor): `[B, T]`

    """
    bs = seq_lens.size(0)
    max_time = seq_lens.max()
    seq_range = torch.arange(0, max_time, dtype=torch.int32, device=seq_lens.device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    mask = seq_range < seq_lens.unsqueeze(-1)
    return mask


def append_sos_eos(ys, sos, eos, pad, device, bwd=False, replace_sos=False):
    """Append <sos> and <eos> and return padded sequences.

    Args:
        ys (list): A list of length `[B]`, which contains a list of size `[L]`
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>

        bwd (bool): reverse ys for backward reference
        replace_sos (bool): replace <sos> with the special token
    Returns:
        ys_in (LongTensor): `[B, L]`
        ys_out (LongTensor): `[B, L]`
        ylens (IntTensor): `[B]`

    """
    _eos = torch.zeros(1, dtype=torch.int64, device=device).fill_(eos)
    ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64),
                    device) for y in ys]
    if replace_sos:
        ylens = np2tensor(np.fromiter([y[1:].size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([y for y in ys], pad)
        ys_out = pad_list([torch.cat([y[1:], _eos], dim=0) for y in ys], pad)
    else:
        _sos = torch.zeros(1, dtype=torch.int64, device=device).fill_(sos)
        ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([torch.cat([_sos, y], dim=0) for y in ys], pad)
        ys_out = pad_list([torch.cat([y, _eos], dim=0) for y in ys], pad)
    return ys_in, ys_out, ylens


def compute_accuracy(logits, ys_ref, pad):
    """Compute teacher-forcing accuracy.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys_ref (LongTensor): `[B, T]`
        pad (int): index for padding
    Returns:
        acc (float): teacher-forcing accuracy

    """
    pad_pred = logits.view(ys_ref.size(0), ys_ref.size(1), logits.size(-1)).argmax(2)
    mask = ys_ref != pad
    numerator = torch.sum(pad_pred.masked_select(mask) == ys_ref.masked_select(mask))
    denominator = torch.sum(mask)
    acc = float(numerator) * 100 / float(denominator)
    return acc


def calculate_cer(seqs_hat, seqs_true):
    """Calculate sentence-level CER score.

    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level CER score
    :rtype float
    """
    char_eds, char_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_chars = seq_hat_text.replace(" ", "")
        ref_chars = seq_true_text.replace(" ", "")
        char_eds.append(editdistance.eval(hyp_chars, ref_chars))
        char_ref_lens.append(len(ref_chars))
    return float(sum(char_eds)) / sum(char_ref_lens)


def calculate_cer_ctc(ys_hat, ys_pad, idx_blank ):
    """Calculate sentence-level CER score for CTC.

    :param torch.Tensor ys_hat: prediction (batch, seqlen)
    :param torch.Tensor ys_pad: reference (batch, seqlen)
    :return: average sentence-level CER score
    :rtype float
    """
    cers, char_ref_lens = [], []
    for i, y in enumerate(ys_hat):
        y_hat = [x[0] for x in groupby(y)]
        y_true = ys_pad[i]
        seq_hat, seq_true = [], []
        for idx in y_hat:
            idx = int(idx)
            if idx != -1 and idx != idx_blank:
                seq_hat.append(str(idx))

        for idx in y_true:
            idx = int(idx)
            if idx != -1 and idx != idx_blank:
                seq_true.append(str(idx))

        hyp_chars = "".join(seq_hat)
        ref_chars = "".join(seq_true)
        if len(ref_chars) > 0:
            cers.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))

    cer_ctc = float(sum(cers)) / sum(char_ref_lens)
    return cer_ctc

def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score.

    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """
    word_eds, word_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)