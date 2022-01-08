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
from vadwebrtc import vadTest

from omegaconf import OmegaConf
import torch
from models.seq2seq.speech2text import Speech2Text

from cffi import FFI


ffi = FFI()
ffi.cdef("""
    const double *fbank_feats_cmvn(int length, char* arg, char* cmvn_path, char* conf_path);
""")
C = ffi.dlopen('./libkaldi-feature.so')

args = OmegaConf.load("aishell/conf.yml")
model = Speech2Text(args)
checkpoint_avg = torch.load('aishell/model', map_location='cpu')
model.load_state_dict(checkpoint_avg['model_state_dict'])
model = model.cuda()
model.eval()
print(" ************ init success ************ ")


class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(300)

    @tornado.web.asynchronous
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
            vaddata = vadTest(1, data)
            pcm_data = ffi.new("char[]", vaddata)

            result = C.fbank_feats_cmvn(len(pcm_data), pcm_data, 'aishell/cmvn.ark'.encode('ascii'), 'aishell/fbank.conf'.encode('ascii'))
            cols = int(result[0])
            rows = int(result[1])

            feat = np.zeros(cols*rows)
            for c in range(cols*rows):
                feat[c] = result[c+2]
            feat = [feat.reshape(-1, rows)]

            aishell = model.decode(feat, args)
            result = {
                "aishell": aishell,
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
