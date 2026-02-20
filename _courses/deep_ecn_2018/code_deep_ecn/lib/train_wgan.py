#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:02:44 2018

Train WGAN

@author: vayer
"""

#%%
import os,sys
#path='/home/vayer/wgw/gwtest-master/code_deep_ecn'
path='./code_deep_ecn/lib'
module_path = os.path.abspath(os.path.join(path))
sys.path.append(module_path)
from wgan import WGAN
from utils import make_keras_picklable,ElapsedTimer,sample_images
import numpy as np

#%%
wgan = WGAN()

#%%
timer = ElapsedTimer()
wgan.train(epochs=30000, batch_size=32, sample_interval=50)
timer.elapsed_time()

#%%
wgan.save_models(path)


