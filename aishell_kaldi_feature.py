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
import pathlib

from omegaconf import OmegaConf
from models.seq2seq.speech2text import Speech2Text
from models.torch_utils import pad_list

import torch

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data
pcm_data = read_wave('aishell/BAC009S0764W0121.wav')

model_path = str(pathlib.Path(__file__).parent)
args = OmegaConf.load(model_path + "/aishell/conf.yml")
args.dict = model_path + '/aishell/dict.txt'

model = Speech2Text(args)
checkpoint = torch.load(model_path + '/aishell/model', map_location='cpu')
model.load_state_dict(checkpoint)

# model = model.cuda()
model.eval()

ffi = FFI()
ffi.cdef("""
    const double *fbank_feats_cmvn(int length, char* arg, char* cmvn_path, char* conf_path);
""")
C = ffi.dlopen(model_path + '/libkaldi-feature.so')

pcm_data = ffi.new("char[]", pcm_data)

result = C.fbank_feats_cmvn(len(pcm_data), pcm_data, model_path + '/aishell/cmvn.ark'.encode('ascii'), model_path + '/aishell/fbank.conf'.encode('ascii'))

cols = int(result[0])
rows = int(result[1])
feat = np.zeros(cols*rows)
for c in range(cols*rows):
    feat[c] = result[c+2]
feat = feat.reshape(cols, rows)

aishell = model.decode([feat], args)
print(aishell)
