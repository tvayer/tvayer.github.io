import matplotlib.pyplot as plt
import pickle
import numpy as np
import os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/cours/deep_ecn_2018/code_deep_ecn/lib'))
sys.path.append(module_path)
from utils import make_keras_picklable
#%%
path='/home/vayer/Documents/code_deep_ecn'
make_keras_picklable()
with open(path+'/generatorwgan.pickle', 'rb') as handle:
    generator = pickle.load(handle)
    
#%%
plot_images(fake=True,generator=generator,noise= np.random.normal(0, 1, (5 * 5, 100)))
