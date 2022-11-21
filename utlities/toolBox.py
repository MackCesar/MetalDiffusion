### System modules
import logging
import os
import random
import sys
import warnings
import argparse

### Memory Management
import gc #Garbage Collector
import time

### Math modules
import numpy as np

### Import tensorflow module, but with supressed warnings to clear up the terminal outputs
# Filter tensorflow version warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

from tensorflow import keras

### Import Stable Diffusion module, Tensorflow version
from ..stable_diffusion_tf.stable_diffusion import StableDiffusion, get_models

print("...Stable Diffusion Tensorflow module loaded...")

import cv2
### Image saving after generation modules
from PIL import Image
from PIL.PngImagePlugin import PngInfo

### WebUi Modules
import gradio as gr

### Misc Modules

### Classes

### Functions

