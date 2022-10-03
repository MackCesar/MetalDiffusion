# STILL UNDER CONSTRUCTION

# Stable Diffusion in TensorFlow / Keras for Apple Metal on Intel Macs

A Keras / Tensorflow implementation of Stable Diffusion designed for Apple Metal on Intel Macs.

The weights were ported from the original implementation.



## Installation

This program is best utilized with python independent from a Mac's base installation and python's virtual environment.

### Install Homebrew if not already

Homebrew is a foundational tool for Mac and is required for this process.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install pyenv

`pyenv` allows you to install Python independently from the Mac's base installation.

```bash
brew install pyenv
```

### Install Python via pyenv

```bash
pyenv install 3.9.0
```

### Installing using the repo

Navigate to a folder you want to create your virtual environment in.

Download the repo, either by downloading the
[zip](https://github.com/divamgupta/stable-diffusion-tensorflow/archive/refs/heads/master.zip)
file or by cloning the repo with git:

```bash
git clone git@github.com:divamgupta/stable-diffusion-tensorflow.git
```

#### Create a virtual environment with *virtualenv*

1) Create your virtual environment for `python3`:

    ```bash
    python3 -m venv venv
    ```
   
2) Activate your virtualenv:

    ```bash
    source venv/bin/activate
    ```

3) Install dependencies using the `requirements.txt` file or the `requirements_m1.txt` file,:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Using the Python interface

If you installed the package, you can use it as follows:

```python
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,
)
img = generator.generate(
    "An astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
)

# for image to image :
img = generator.generate(
    "A Halloween bedroom",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image="/path/to/img.png"
)


Image.fromarray(img[0]).save("output.png")
```

### Using `text2image.py` from the git repo

Assuming you have installed the required packages, 
you can generate images from a text prompt using:

```bash
python text2image.py --prompt="An astronaut riding a horse"
```

The generated image will be named `output.png` on the root of the repo.
If you want to use a different name, use the `--output` flag.

```bash
python text2image.py --prompt="An astronaut riding a horse" --output="my_image.png"
```

Check out the `text2image.py` file for more options, including image size, number of steps, etc.

## Example outputs 

The following outputs have been generated using this implementation:

1) *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](https://user-images.githubusercontent.com/1890549/190841598-3d0b9bd1-d679-4c8d-bd5e-b1e24397b5c8.png)


2) *Spider-Gwen Gwen-Stacy Skyscraper Pink White Pink-White Spiderman Photo-realistic 4K*

![a](https://user-images.githubusercontent.com/1890549/190841999-689c9c38-ece4-46a0-ad85-f459ec64c5b8.png)


3) *A vision of paradise, Unreal Engine*

![a](https://user-images.githubusercontent.com/1890549/190841886-239406ea-72cb-4570-8f4c-fcd074a7ad7f.png)


## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
3) https://github.com/divamgupta/stable-diffusion-tensorflow
