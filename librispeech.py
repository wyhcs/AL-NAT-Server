import time
import os
import re
import copy
import numpy as np
import contextlib
import wave
from cffi import FFI

from omegaconf import OmegaConf
import torch
from models.seq2seq.speech2text import Speech2Text
from models.torch_utils import pad_list

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data

pcm_data = read_wave('librispeech/367-130732-0000.wav')
args = OmegaConf.load("librispeech/conf.yml")
model = Speech2Text(args)
checkpoint_avg = torch.load('librispeech/model', map_location='cpu')
model.load_state_dict(checkpoint_avg['model_state_dict'])

model = model.cuda()
model.eval()

ffi = FFI()
ffi.cdef("""
    const double *fbank_feats_cmvn(int length, char* arg, char* cmvn_path, char* conf_path);
""")
C = ffi.dlopen('./libkaldi-feature.so')


pcm_data = ffi.new("char[]", pcm_data)

start = time.clock()

result = C.fbank_feats_cmvn(len(pcm_data), pcm_data, 'librispeech/cmvn.ark'.encode('ascii'), 'librispeech/fbank.conf'.encode('ascii'))

cols = int(result[0])
rows = int(result[1])
feat = np.zeros(cols*rows)
for c in range(cols*rows):
    feat[c] = result[c+2]
feat = [feat.reshape(cols, rows)]

i = 1
for x in range(1):
    librispeech = model.decode(feat, args)
    print(librispeech)
    i += 1
    print(i)

end = time.clock()
t=end-start
print("Runtime is ：",t)
