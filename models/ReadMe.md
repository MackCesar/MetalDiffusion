# Weights (aka models)

Currently, this implementation of Stable Diffusion accepts weights from the following formats:

1) Keras ".h5" format
2) Pytorch ".ckpt" format

Place your ".ckpt" within this models folder.

## Keras ".h5"

The program searches the models folder for another folder that has the model name. For example:

`StableDiffusion2/`

Within that folder, the program is looking for these four ".h5" files:

1) decoder.h5
2) diffusion_model.h5
3) encoder.h5
4) text_encoder.h5

As of 3/14/23, ".h5" formats for Stable Diffusion are uncommon. This program can, however, convert ".ckpt" to ".h5".

The benefits of using a ".h5" format are speed; TensorFlow Keras can load ".h5" weights faster than ".ckpt"

## Pytorch ".ckpt"

The program searches the models folder for files that end in ".ckpt".

## "VAE/" Folder

Place VAE ".ckpt"s here.

Currently, as of 3/14/23, the program can only load VAE's as ".ckpt"

## "embeddings/" Folder

Place Textual Inversion weights, aka Text Embeddings, here.

Currently, as of 3/14/23, the program accepts weights in the ".pt" and ".bin" formats.

## "controlnets/" Folder

Place ControlNet weights here.

Currently, as of 5/10/23, the program accepts weights in the ".safetensors" and ".pth" formats.
