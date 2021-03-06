#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:44:54 2018

Train GAN

@author: vayer
"""

#%%
import os,sys
#path='/home/vayer/wgw/gwtest-master/code_deep_ecn'
path='./code_deep_ecn/lib'
#path='/Users/vayer/Documents/cours/deep_ecn_2018/code_deep_ecn/lib'
module_path = os.path.abspath(os.path.join(path))
sys.path.append(module_path)
from gan import GAN
from utils import make_keras_picklable,ElapsedTimer,sample_images
import numpy as np


#%%
gan = GAN()

#%%

make_keras_picklable()

timer = ElapsedTimer()
gan.train(epochs=20, batch_size=32, sample_interval=50)
timer.elapsed_time()
gan.save_models(path)
