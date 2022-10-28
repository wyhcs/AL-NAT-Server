#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import torch

for x in range(200):
	url = "http://183.175.12.88:8080/asr"
	files = {'wave':open('librispeech/61-70968-0000.wav','rb')}
	reponse = requests.post(url, files=files)
	text = eval(reponse.text)
	print(text)

