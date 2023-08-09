# MetalDiffusion
## Stable Diffusion in TensorFlow / Keras for Apple Metal on Intel Macs

A Keras / Tensorflow implementation of Stable Diffusion, intended for Apple Metal on Intel Macs. This implementation will utilize GPU's on Intel Macs.

The weights were ported from the original implementation of Stable Diffusion.



## Installation

Currently, MetalDiffusion can only be installed via the Terminal App. Follow the instructions here:

https://github.com/soten355/MetalDiffusion/wiki/Installation

## Usage

After installation, MetalDiffusion is easy to use. Follow the steps here:

https://github.com/soten355/MetalDiffusion/wiki/Starting-MetalDiffusion

## Wiki

Further details about MetalDiffusion, including on how to use Text Embeddings (textural inversion), LoRA's, and more can be found in the Github wiki:

https://github.com/soten355/MetalDiffusion/wiki

## Example outputs 

The following outputs have been generated using this implementation:

1) **Stable Diffusion 1.5**: *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](creations/7771775831-SD1p5.png)


2) ***OpenJourney***: *mdjrny-v4 style, Cookie Monster as a king, portrait, renaissance style, Rembrandt, oil painting, chiaroscuro, highly detailed, textured, king*

![a](creations/965345875.01.png)


3) ***Stable Diffusion 2.1***: *A vision of paradise, Unreal Engine*

![a](creations/14804316391.png)

4) ***OpenJourney***: Video example:

[Video File](creations/videoExample.mp4)

5) ***DreamShaper*** with <ghst-3000> text embedding: *Grainy portrait of a space traveller, at night in a busy spaceport, highly detailed, cinematic lighting, moody, neon lights, exterior, stars in the sky, ghst-3000*

![a](creations/4530009741.png)

6) ***ControlNet HED*** with Pre-Processed Example: *tintype photograph of Sigourney Weaver smiling, her hand holding an orange tabby cat, vintage black and white photography, wet plate photography, ambrotype photograph, daguerreotype photo, science fiction, highly detailed, intricate details*

<img src="creations/20213827041.png" width="256" />
<img src="creations/20213827041_HED.png" width="256" />

## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
3) https://github.com/divamgupta/stable-diffusion-tensorflow
