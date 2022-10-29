#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pathlib

from omegaconf import OmegaConf
from models.seq2seq.speech2text import Speech2Text
from models.torch_utils import pad_list

import torch
import torchaudio

model_path = str(pathlib.Path(__file__).parent)
args = OmegaConf.load(model_path + "/aishell/conf.yml")
args.dict = model_path + '/aishell/dict.txt'

model = Speech2Text(args)
checkpoint = torch.load(model_path + '/aishell/model', map_location='cpu')
model.load_state_dict(checkpoint)

model = model.cuda()
model.eval()

sound, sample_rate = torchaudio.load(model_path + '/aishell/BAC009S0764W0121.wav')
feat = torchaudio.compliance.kaldi.fbank(sound, num_mel_bins=80, window_type='hamming').numpy()
feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

aishell = model.decode([feat], args)
print(aishell)
