# Weights (aka models)

MetalDiffusion accepts Stable Diffusion weights in the following formats:

1) Diffusers format
2) Keras ".h5" format
3) HuggingFace Safetensors ".safetensors" format
4) Pytorch ".ckpt" format

# Place weights specific to the format in the following folders:


## models/ckpt - Pytorch ".ckpt"

The program searches the models/ckpt folder for files that end in ".ckpt".

## models/controlnets - ControlNet

Place ControlNet weights here.

Currently, as of 8/5/23, the program accepts weights in the ".safetensors", ".pth", and HuggingFace formats.

When using the __Diffusers Render Engine__, the program will only work with HuggingFace formatted folders.

## models/diffusers - Diffusers/HuggingFace Format

The program searches for weights as a folder in this folder. For example, if a weight is named "MyWeight", it will be here as a folder: "models/diffusers/MyWeight".

You will need to have HuggingFace specific folders to work with MetalDiffusion. For an example, look here: https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

MetalDiffusion can convert ".safetensors" and ".ckpt" to the HuggingFace format. Those conversions are saved in this folder.

## models/embeddings - Textural Inversion weights ".pt" and ".bin"

Place Textual Inversion weights, aka Text Embeddings, here.

Currently, as of 3/14/23, the program accepts weights in the ".pt" and ".bin" formats.

## models/LoRA - LoRA's

Place LoRA weights here.

## models/safetensors - ".safetensors"

Place ".safetensors" formatted weights here. These weights can be used by both render engines and is the preferred method of sharing weights.

## models/tensorflow - TensorFlow/Keras ".h5"

The program searches the models folder for another folder that has the model name. For example:

`StableDiffusion2/`

Within that folder, the program is looking for these four ".h5" files:

1) decoder.h5
2) diffusion_model.h5
3) encoder.h5
4) text_encoder.h5

As of 3/14/23, ".h5" formats for Stable Diffusion are uncommon. This program can, however, convert ".ckpt" to ".h5".

The benefits of using a ".h5" format are speed; TensorFlow Keras can load ".h5" weights faster than ".ckpt"

## "VAE/" Folder

Place VAE ".ckpt"s here.
