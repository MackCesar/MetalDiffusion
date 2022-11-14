# Stable Diffusion in TensorFlow / Keras for Apple Metal on Intel Macs

A Keras / Tensorflow implementation of Stable Diffusion, intended for Apple Metal on Intel Macs. This implementation will utilize GPU's on Intel Macs.

The weights were ported from the original implementation of Stable Diffusion.



## Installation

This program is best utilized within a python virtual environment, making it independent of the python already installed on MacOS. These installation instructions are geared towards use on an Intel Mac.

### Programs required

1) Terminal
2) Homebrew
3) Pyenv
4) Python
5) Git

`Terminal` will be used for all of the following commands:

### Install Homebrew, if not already installed

`Homebrew` is a fundamental tool for Mac and is required for this installion process.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install/Update pyenv

`pyenv` allows you to install Python independently from the Mac's base installation.

```bash
brew install pyenv
```

If pyenv is already installed, then Homebrew will update pyenv

### Install Python via pyenv

```bash
pyenv install 3.9.0
```

### Set the global Python

Set the global python that the Mac will use to the newly installed, via pyenv, python.

```bash
pyenv global 3.9.0
```

This step will override which version and the location of python the Mac will use, allowing for easier updating and control.

### Install/Update git

`git` allows you to download this repository (repo)

```bash
brew install git
```

If git is already installed, then Homebrew will update git

### Download the github repo

Navigate to a folder/directory you want to create your virtual environment in. For example: `/Users/MacUser`

Download the repo, either by downloading the
[zip](https://github.com/soten355/stable-diffusion-tensorflow-IntelMetal/archive/refs/heads/master.zip)
file or by cloning the repo with git:

```bash
git clone https://github.com/soten355/stable-diffusion-tensorflow-IntelMetal.git
```

### Start working in the virutal environment

First, navigate to the root folder in the github repo that was just installed. For example:

```bash
cd /Users/MacUser/stable-diffusion-tensorflow-IntelMetal/
```

#### Create a virtual environment with *virtualenv*

1) Create your virtual environment for `python`:

    ```bash
    python -m venv venv
    ```
   
2) Activate your virtual environment:

    ```bash
    source venv/bin/activate
    ```

3) Install dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4) When finished using Stable Diffusion, deactivate the virtual environment:
    
    ```bash
    deactivate
    ```

## Usage

### As a Web UI

For a user-friendly interface, use the web UI made with Gradio. With `Terminal` run this command:

```bash
python dreamWebUI.py
```

The web UI will automatically load into your preferred browser.

To finish the program, in `Terminal` type `CTRL+C` and the program will quit or quit `Terminal` entirely.

### As a Python module

You can use the repo as a python module in a custom script:

```python
### Import Stable Diffusion Tensorflow
from stable_diffusion_tf.stable_diffusion import StableDiffusion

### Import image creation module
from PIL import Image

### Create a class
generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,
)

### Use the generate function in the class to create an image. It will return an array which can be converted into an iamge
img = generator.generate(
    "An astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
)

# Or also include an image for image to image
img = generator.generate(
    "A Halloween bedroom",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image="/path/to/img.png"
)

### Convert the returned array to an actual image and save it
Image.fromarray(img[0]).save("output.png")
```

### Using `text2image.py` in `Terminal`

*text2image.py* is a pre-created python script that can be used from the command line within the virtual environment:

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
