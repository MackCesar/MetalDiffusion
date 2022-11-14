print("\n...starting program...\n\n...loading modules...")
### System modules
import logging
import os
import random
import sys
import warnings

print("\n...system modules loaded...")

### Math modules
import numpy as np

print("...math modules loaded...")

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

print("...tensorflow module loaded...")

### Import Stable Diffusion module, Tensorflow version
from stable_diffusion_tf.stable_diffusion import StableDiffusion, get_models

print("...Stable Diffusion Tensorflow module loaded...")

import cv2
### Image saving after generation modules
from PIL import Image
from PIL.PngImagePlugin import PngInfo

print("...image modules loaded...")

### WebUi Modules
import gradio as gr

print("...WebUI module loaded...")

### Misc Modules
import utilities.deviceTest as deviceTest
import utilities.modelFinder as mf
import utilities.settingsControl as settingsControl
import utilities.readWriteFile as readWriteFile

print("...all modules loaded!")

### Global Variables
print("\nCreating global variables...")
finalImage = ()
programStarted = False
publicText = "Welcome!"
result = None
program = None
generator = None
model = None
# Custom settings - Factory defaults
stepsMax = 64
scaleMax = 20
batchMax = 8
defaultBatchSize = 4
modelsLocation = "models"
defaultModel = "mdjrny-v4.ckpt"
maxMemory = 229376
creationLocation = "creations/"
# Try loading custom settings from user file, otherwise continue with factory settings
try:
    userSettings = settingsControl.loadSettings("userData/userPreferences.txt")
    stepsMax, scaleMax, batchMax, defaultBatchSize, modelsLocation, defaultModel, maxMemory, creationLocation = [userSettings[i] for i in range(8)]
except Exception as e:
    print("Factory defaults loaded!\nCreating new preferences file.")
    print(e)
    settingsControl.createUserPreferences(
        "userData/userPreferences.txt",
        [
            stepsMax,
            scaleMax,
            batchMax,
            defaultBatchSize,
            modelsLocation,
            defaultModel,
            maxMemory,
            creationLocation
        ]
    )

## Colors (part of global variables)
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

## Prompt Settings
try:
    starterPrompt = settingsControl.loadSettings("userData/promptGenerator.txt", 1)
except Exception as e:
    print(e)
    starterPrompt = []

print("...global variables created!")

### Functions

def publicPrint(text):
    print(text)
    return text

def generateArt(prompt="dinosaur skateboarding", height=512, width=512, scale=7.5, steps=32, seed=None, input_image=None, input_image_strength = 0.5, negativePrompt = None, pytorchModel = None, batchSize = 1, saveSettings = True):

    ### Start stable Diffusion
    # publicText = publicPrint("\nStarting Stable Diffusion with Tensor flow and Apple Metal.")

    global programStarted
    global generator
    global model
    global defaultModel
    global maxMemory

    if (deviceTest.getTensorMemorySize(batchSize, width, height, 4) > int(maxMemory)):
        print(deviceTest.getTensorMemorySize(batchSize, width, height, 4))
        print("\ntoo much memory!\n")

    if programStarted is True:
        print(color.BLUE, color.BOLD, "\nRe-starting Stable Diffusion with Tensor flow and Apple Metal.", color.END)
        # Try to clear memory
        tf.keras.backend.clear_session()

        #Check our model

        #if pytorchModel == defaultModel: # Use the tensorflow model
        #    print("\nUpdating with default model\n")
        #    downloadWeights = True
        #else: # Use pytorch model, which will be called later
        downloadWeights = False

        if height != generator.img_height or width != generator.img_width:
            print(color.CYAN,color.BOLD,"Re-compiling Stable Diffusion models",color.END,"\n")
            updateGenerator(height, width, downloadWeights)

    else:
        print(color.BLUE, color.BOLD, "\nStarting Stable Diffusion with Tensor flow and Apple Metal.", color.END)
        programStarted = True

        # Which Mdoel to use?

        if pytorchModel == "Stable Diffusion 1.4":
            downloadWeights = True
            model = "Stable Diffusion 1.4"
        else:
            downloadWeights = False
        
        print(color.CYAN,color.BOLD,"Loading Metal, connecting to GPU, and compiling Stable Diffusion models",color.END,"\n")

        # Create generator with StableDiffusion class
        generator = StableDiffusion(
            img_height=height,
            img_width=width,
            jit_compile=False,
            download_weights=downloadWeights
        )

    # Are we using a pytroch model? If downloadWeights is fale, then yes we are!
    if pytorchModel != "Stable Diffusion 1.4":
        if pytorchModel != model:
            modelLocation = "models/" + pytorchModel
            generator.load_weights_from_pytorch_ckpt(modelLocation)
            model = pytorchModel
    
    # Set seed if not given
    if seed is None or 0:
        seed = random.randint(1000, sys.maxsize)

    print(color.PURPLE, "\nGenerating image", color.END)

    # Use the generator function within the newly created class to generate an array that will become an image
    imgs = generator.generate(
        prompt,
        num_steps = steps,
        unconditional_guidance_scale = scale,
        temperature = 1,
        batch_size = batchSize,
        seed = seed,
        input_image = input_image,
        input_image_strength = input_image_strength,
        negativePrompt = negativePrompt
    )

    print(color.BOLD, color.GREEN, "\nFinished!\n")

    ### Create final image from the generated array ###

    # Generate PNG metadata for reference
    metaData = PngInfo()
    metaData.add_text('prompt', prompt)

    # Save settings
    if saveSettings is True:
        readWriteFile.writeToFile("creations/" + str(seed) + ".txt", [prompt, negativePrompt, width, height, scale, steps, seed, pytorchModel, batchSize, input_image_strength])

    # Multiple Image result:
    for img in imgs:
        print("Processing image!")
        imageFromBatch = Image.fromarray(img)
        imageFromBatch.save(creationLocation + str(seed) + str(batchSize) + ".png", pnginfo = metaData)
        print("Image saved!\n")
        batchSize = batchSize - 1

    print("Returning images!",color.END)
    return imgs

def createPromptComponents(variable):

    totalComponents = []
    for key in variable:
        component = gr.Dropdown(
            choices = variable[key],
            label = str(key),
            value = None
        )

        totalComponents.append(component)

    return totalComponents

def randomSeed():
    newSeed = random.randint(0, 2 ** 31)
    return newSeed

def updateGenerator(height, width, downloadWeights):
    print("Updating generator...")
    # Did any initial settings change for the generator?
    global generator
    global defaultModel

    # newDimensions
    newHeight = None
    newWidth = None
    
    # Different height?
    if generator.img_height is not height:
        newHeight = height
        print("...new height detected...")
    
    if generator.img_width is not width:
        print("...new width detected...")
        newWidth = width
    
    # Fill the empty newDimensions with the original dimensions
    if newHeight is None:
        newHeight = generator.img_height
    if newWidth is None:
        newWidth = generator.img_width
    
    generator.img_height = newHeight
    generator.img_width = newWidth

    if model == defaultModel:
        downloadWeights = True

    # Compile new dimensions
    print("...applying new changes to generator.")
    text_encoder, diffusion_model, decoder, encoder = get_models(generator.img_height, generator.img_width, download_weights=downloadWeights)
    generator.text_encoder = text_encoder
    generator.diffusion_model = diffusion_model
    generator.decoder = decoder
    generator.encoder = encoder

    if model != defaultModel:
        print("...updating pytorch model...")
        modelLocation = "models/" + model
        generator.load_weights_from_pytorch_ckpt(modelLocation)

    print("Changes to generator completed!")

def addToPrompt(originalPrompt, slotA, slotB, slotC, slotD, slotE):
    # Combine slots into a list (because we couldn't pass a list of gradio components)
    additionList = [slotA, slotB, slotC, slotD, slotE]
    addition = ""
    for item in additionList:
        if item != "":
            if addition == "":
                addition = str(item)
            else:
                addition = addition + ", " + str(item)
    
    if originalPrompt == "" or None:
        newPrompt = str(addition)
    else:
        newPrompt = str(originalPrompt) + ", " + str(addition)

    return newPrompt, gr.update(value = ""), gr.update(value = ""), gr.update(value = ""), gr.update(value = ""), gr.update(value = "")


### Main Web UI Layout ###

## Define components outside of gradio's interface so they can be accessed regardless of child/parent position in the layout

# Prompts
prompt = gr.Textbox(
    label = "Prompt - What should the AI create?"
)

negativePrompt = gr.Textbox(
    label = "Negative Prompt - What should the AI avoid when creating?"
)

# Steps
steps = gr.Slider(
    minimum = 2,
    maximum = int(stepsMax),
    value = int(stepsMax) / 2,
    step = 1,
    label = "Steps - How many times the AI should sample"
)

# Scale
scale = gr.Slider(
    minimum = 2,
    maximum = int(scaleMax),
    value = 7.5,
    step = 0.1,
    label = "Guidance Scale - How closely should the AI follow the prompt"
)

# Height
height = gr.Dropdown(
    choices = [1,2,4,8,16,32,64,128,256,384,512,768,896,1024],
    value = 512,
    label = "Height - How tall should the final image be?"
)

# Width
width = gr.Dropdown(
    choices = [1,2,4,8,16,32,64,128,256,384,512,768,896,1024],
    value = 512,
    label = "Width - How wide should the final image be?"
)

# Seed
seed = gr.Number(
    value = random.randint(0, 2 ** 31),
    label = "Seed  - Unique number for the image created",

)

# Input Image

inputImage = gr.Image(
    label = "Input Image"
)

# Input Image Strength
inputImageStrength = gr.Slider(
    minimum = 0,
    maximum = 1,
    value = 0.5,
    step = 0.1,
    label = "Input Image Strength - 0 = Don't change the image, 1 = ignore image entirely"
)

# Start Button
startButton = gr.Button("Start")

startButton.style(
    full_width = True
)

# Models/weights

modelsWeights = mf.findModels(modelsLocation, ".ckpt")

listOfModels = gr.Dropdown(
            choices = modelsWeights,
            label = "Model",
            value = defaultModel
        )

consoleLog = gr.Markdown("Welcome!")

# Batch Size

batchSizeSelect = gr.Slider(
    minimum = 1,
    maximum = int(batchMax),
    value = int(defaultBatchSize),
    step = 1,
    label = "Batch Size - How many results to make?"
)

# Resulting Image(s)

result = gr.Gallery(
    label = "Results",
)

result.style(
    grid = 3,
    height = 512
)

# Prompt Engineering

starterPrompts = createPromptComponents(starterPrompt)

addPrompt = gr.Button("Add to prompt")

importPromptLocation = gr.File(
    label = "Import Prior Prompt and settings for prompt",
    type = "file"
)

importPromptButton = gr.Button("Import prompt")

# User Settings

saveSettings = gr.Checkbox(
    label = "Save settings used for prompt creation?",
    value = True
)

## Main Layout

with gr.Blocks(
    title = "Stable Diffusion"
) as demo:
    #Title
    gr.Markdown(
        "<center><span style = 'font-size: 32px'>Stable Diffusion</span><br><span style = 'font-size: 16px'>Tensorflow and Apple Metal<br>Intel Mac</span></center>"
    )

    with gr.Row():
        with gr.Column(
            scale = 3,
            variant = "panel"
        ):
            # Prompts
            prompt.render()
            negativePrompt.render()
        with gr.Column(
            scale = 1,
            variant = "compact"
        ):

            # Start Button
            startButton.render()

    # Image to Image
    with gr.Row():
        with gr.Column():

            with gr.Tab("Text to Image"):
                with gr.Row():
                    # Basic Settings
                    with gr.Column():
                        gr.Markdown("<center><b>Basic Settings</b></center>Necessary options")

                        # Width
                        width.render()

                        # Height
                        height.render()

                        # Batch Size
                        batchSizeSelect.render()
                    
                    # Elementary settings
                    with gr.Column():
                        gr.Markdown("<center><b>Elementary Settings</b></center>For more control")

                        # Steps
                        steps.render()

                        # Scale
                        scale.render()

                        # Seed
                        seed.render()

                        newSeed = gr.Button("New Seed")

            with gr.Tab("Input Image"):
                gr.Markdown("Feed a starting image into the AI to give it inspiration")

                inputImage.render()

                # Input Image Strength

                inputImageStrength.render()

            with gr.Tab("Prompt Generator"):
                gr.Markdown("Tools to generate useful prompts")
                
                # Starter Prompts
                for item in starterPrompts:
                    item.render()

                addPrompt.render()

            with gr.Tab("Import"):
                # Import prior prompt and settings
                gr.Markdown("Import prior prompt and generator settings")
                importPromptLocation.render()
                importPromptButton.render()

            with gr.Tab("Advanced Settings"):
                # Model Selection
                listOfModels.render()

                # Save settings used for creation?
                saveSettings.render()

        with gr.Column():
            # Result
            with gr.Column():
                gr.Markdown("<center><span style = 'font-size: 24px'><b>Result</b></span></center>")
                
                result.render()

    # Event actions
    startButton.click(
        fn = generateArt,
        inputs = [prompt, height, width, scale, steps, seed, inputImage, inputImageStrength, negativePrompt, listOfModels, batchSizeSelect, saveSettings],
        outputs = result
    )

    newSeed.click(
        fn = randomSeed,
        inputs = None,
        outputs = seed,
    )

    addPrompt.click(
        fn = addToPrompt,
        inputs = [prompt, starterPrompts[0], starterPrompts[1], starterPrompts[2], starterPrompts[3], starterPrompts[4]],
        outputs = [prompt, starterPrompts[0], starterPrompts[1], starterPrompts[2], starterPrompts[3], starterPrompts[4]]
    )

    importPromptButton.click(
        fn = readWriteFile.readFromFile,
        inputs = importPromptLocation,
        outputs = [prompt, negativePrompt, width, height, scale, steps, seed, listOfModels, batchSizeSelect, inputImageStrength,]
    )

print("\nLaunching Gradio\n")

demo.launch(
    inbrowser = True,
    show_error = True
)
