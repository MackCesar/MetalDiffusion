### System modules
import argparse
import os
import warnings
import logging
import random
import sys

### Math modules
import numpy as np
import math
import random

### Tensorflow modules

import tensorflow as tf
try:
   from keras import backend as K
except Exception as e:
   print(e)
   from tensorflow.keras import backend as K

### Colors
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

### Main Functions

def getModelMemoryUsage(batch_size, model):

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count

    return gbytes

def timestep_embedding(timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))

def getTensorMemorySize(batchSize, width, height, dType):
    memoryUsage = batchSize * width * height * 4
    memoryUsage = memoryUsage * dType
    return(memoryUsage)

def listDevices():
    devices = tf.config.list_physical_devices()
    return devices

print(color.BLUE, color.BOLD, "\nStarting device test for tensorflow\n",color.END)

batchSize = 2
width = 512 // 8
height = 1024 // 8
dType = 32 // 8

maxMemory = 327680

print(
    getTensorMemorySize(batchSize, width, height, dType)
    )
