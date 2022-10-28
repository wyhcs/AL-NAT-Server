#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np

from omegaconf import OmegaConf
from models.seq2seq.speech2text import Speech2Text
from models.torch_utils import pad_list

import torch
import torchaudio

model_path = str(pathlib.Path(__file__).parent)

args = OmegaConf.load(model_path + "/librispeech/conf.yml")
args.dict = model_path + '/librispeech/dict.txt'
args.wp_model = model_path + '/librispeech/wp.model'
args.kenlm_path = model_path + '/librispeech/3-gram.klm'

model = Speech2Text(args)
checkpoint = torch.load(model_path + '/librispeech/model', map_location='cpu')
model.load_state_dict(checkpoint)

model = model.cuda()
model.eval()


sound, sample_rate = torchaudio.load(model_path + '/librispeech/61-70968-0000.wav')
feat = torchaudio.compliance.kaldi.fbank(sound, num_mel_bins=80, window_type='hamming').numpy()
feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

librispeech = model.decode([feat], args)
print(librispeech)
