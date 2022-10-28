#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import time
import os
import re
import copy
import contextlib
import wave
from cffi import FFI
import numpy as np

from omegaconf import OmegaConf
from models.seq2seq.speech2text import Speech2Text
from models.torch_utils import pad_list

import torch
import torchaudio

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data


pcm_data = read_wave('aishell/BAC009S0764W0121.wav')

args = OmegaConf.load("aishell/conf.yml")
model = Speech2Text(args)
checkpoint_avg = torch.load('aishell/model', map_location='cpu')
model.load_state_dict(checkpoint_avg['model_state_dict'])

model = model.cuda()
model.eval()

ffi = FFI()
ffi.cdef("""
    const double *fbank_feats_cmvn(int length, char* arg, char* cmvn_path, char* conf_path);
""")
C = ffi.dlopen('./libkaldi-feature.so')

pcm_data = ffi.new("char[]", pcm_data)

result = C.fbank_feats_cmvn(len(pcm_data), pcm_data, 'aishell/cmvn.ark'.encode('ascii'), 'aishell/fbank.conf'.encode('ascii'))

cols = int(result[0])
rows = int(result[1])
feat = np.zeros(cols*rows)
for c in range(cols*rows):
    feat[c] = result[c+2]
feat = feat.reshape(cols, rows)

# sound, sample_rate = torchaudio.load('aishell/BAC009S0764W0121.wav')
# feat = torchaudio.compliance.kaldi.fbank(sound, num_mel_bins=80, window_type='hamming').numpy()
# feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

aishell = model.decode([feat], args)
print(aishell)
