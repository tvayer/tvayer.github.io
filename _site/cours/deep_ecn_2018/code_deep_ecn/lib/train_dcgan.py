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
#path='/Users/vayer/Documents/cours/deep_ecn_2018/code_deep_ecn/lib'
path='./code_deep_ecn/lib'
module_path = os.path.abspath(os.path.join(path))
sys.path.append(module_path)
from dcgan import DCGAN
from utils import make_keras_picklable,ElapsedTimer,sample_images
import numpy as np

#%%
mnist_dcgan = DCGAN()

#%%
timer = ElapsedTimer()
mnist_dcgan.train(epochs=10000, batch_size=256, save_interval=500)
timer.elapsed_time()
mnist_dcgan.save_models(path)
#%%

sample_images(generator=mnist_dcgan.generator,noise= np.random.normal(0, 1, (5 * 5, 100)))



#%%
mnist_dcgan.save_models(path)