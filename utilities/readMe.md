# Utilities

## modelFinder.py
**Model Finder** is a script that locates the weights for models:

1) VAE - .ckpt
2) Diffusion - .ckpt, .h5 (as a folder)
3) Text Embeddings/Inversion - .pt, .bin

## readWriteFile.py
**Read and Write to File** is a script that writes data into .txt files.

Currently, it is used to write:
1) the art/cinema creation settings of a generation. The input text is a list of strings that are written into a file. Every index is a line of data
2) the analysis of a .ckpt file. This was used in the development of SD2.x adoption

## settingsControl.py
**User Settings Control** is a script that reads and writes the user preferences and settings.

This script contains the factory defaults for the webUI.

## videoUtilities.py
**Video Utilities** is a script that handles all things regarding video creation.

Currently, the zoom out function isn't working as intended. Beware!

## controlNetUtilities.py
**ControlNet Utilities** focuses on pre-processing images for the ControlNet model.

Currently, it can only pre-process images with Canny Edge detection and Soft Edge detection. The reason: other pre-processing methods currently need PyTorch and my aim with the program is to be agnostic of PyTorch and TensorFlow.

NOTE: You don't need to pre-process the image in MetalDiffusion, you can bypass it entirely.
