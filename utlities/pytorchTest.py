#System modules
import argparse
import os
import warnings
import logging
import random
import sys

#Modules for coloring text in the console
#from colorama import Fore, Back, Style

# import tensorflow, but with supressed warnings
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

# Import Stable Diffusion module, Tensorflow version
from stable_diffusion_tf.stable_diffusion import StableDiffusion

# Image saving after generation modules
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="A beautiful street view, artstation concept art",
    help="the prompt to render",
)

parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default="output.png",
    help="where to save the output image",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixels",
)

parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixels",
)

parser.add_argument(
    "--aspect_ratio",
    type=float,
    nargs="?",
    help="Aspect ratio of the final image",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--steps", type=int, default=50, help="number of ddim sampling steps"
)

parser.add_argument(
    "--seed",
    type=int,
    help="optionally specify a seed integer for reproducible results",
)

parser.add_argument(
    "--mp",
    default=False,
    action="store_true",
    help="Enable mixed precision (fp16 computation)",
)

parser.add_argument(
    "--input_image",
    type=str,
    nargs="?",
    help="Input image used for Image to Image",
)

args = parser.parse_args()

if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

if (args.H > 896):
    print(color.RED,"Height too tall! Maximum is 896 pixels. Reducing your height input to 896 pixels",color.END)
    args.H = 896

if (args.W > 896):
    print(color.RED,"Height too tall! Maximum is 896 pixels. Reducing your height input to 896 pixels",color.END)
    args.W = 896


# Create a random seed if one is not provided
if args.seed is None:
    args.seed = random.randint(1000, sys.maxsize)

### Start stable Diffusion ###
print(color.BLUE, color.BOLD, "\nStarting Stable Diffusion with Tensor flow and Apple Metal.\n", color.END)
print(color.CYAN, color.UNDERLINE,"Loading Metal, connecting to GPU, and compiling Stable Diffusion models",color.END,"\n")

generator = StableDiffusion(
    img_height=args.H,
    img_width=args.W,
    jit_compile=False,
)

print(color.YELLOW, "\nLoading pytorch weights\n", color.END)

# diffusion_model_weights = keras.utils.get_file(
#        origin="https://huggingface.co/ogkalu/Comic-Diffusion/resolve/main/comic-diffusion.ckpt",
#        file_hash="33789685ab6488d34e6310f7e6da5c981194ce59ef4b6890f681d5cc5b9c62cc",
#    )

diffusion_model_weights = "pytorchModels/jinxmerge2.ckpt"

generator.load_weights_from_pytorch_ckpt(diffusion_model_weights)

print(color.PURPLE, "\nGenerating image", color.END)

img = generator.generate(
    args.prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=args.scale,
    temperature=1,
    batch_size=1,
    seed=args.seed,
    input_image=args.input_image,
)

print(color.BOLD, color.GREEN, "\nFinished!\n", color.END)

### Create the fianl image ###

# Generate PNG metadata for reference
metaData = PngInfo()
metaData.add_text('prompt', args.prompt)

# Save final image. File name is the seed and prompt is saved into the metadata
finalImage = Image.fromarray(img[0])

finalImage.save(str(args.seed)+".png", pnginfo=metaData)

print(color.GREEN,"Image saved!\n",color.END)