#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import torch

for x in range(200):
	url = "http://127.0.0.1:9009/asr"
	files = {'wave':open('aishell/BAC009S0764W0121.wav','rb')}
	reponse = requests.post(url, files=files)
	text = eval(reponse.text)
	print(text)

