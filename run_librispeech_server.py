#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import re
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.log
import tornado.httpclient
from tornado.options import options, define
from tornado.escape import json_encode, json_decode, utf8
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import json
import logging
import uuid
import pathlib

from omegaconf import OmegaConf
import torch
import torchaudio
from models.seq2seq.speech2text import Speech2Text

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
print(" ************ init success ************ ")


class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(300)

    @tornado.gen.coroutine
    def post(self):
        try:
            begintime = time.time()
            data = self.request.files['wave'][0]['body']
        except:
            self.write("Cannot find input wave data")
            return

        json_result = yield self.block_task(data)
        self.write(json_result)

    @run_on_executor
    def block_task(self, data):
        try:
            wav_data = np.fromstring(data,dtype=np.int16)
            wav_data = wav_data*1.0/(max(abs(wav_data)))
            wav_data =  torch.from_numpy(wav_data).unsqueeze(0) 

            feat = torchaudio.compliance.kaldi.fbank(wav_data, num_mel_bins=80, window_type='hamming').numpy()
            feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

            librispeech = model.decode([feat], args)
            result = {
                "librispeech": librispeech,
            }
            return json.dumps(result,ensure_ascii=False)
        except tornado.gen.BadYieldError as e:
            pass

def make_app():
    return tornado.web.Application([
        (r"/asr", MainHandler),
    ], autoreload=False, debug=False)

def main():
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.bind(8080)
    server.start(1)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
   main()
