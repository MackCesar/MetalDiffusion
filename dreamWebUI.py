print("\n...starting program...\n\n...loading modules...")
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
import utilities.modelFinder as mf
import utilities.settingsControl as settingsControl
import utilities.readWriteFile as readWriteFile
import utilities.videoUtilities as videoUtil

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
modelsLocation = "models/"
defaultModel = "samdoesarts v2.ckpt"
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

def addToPrompt(originalPrompt, slotA, slotB, slotC, slotD, slotE):
    # Combine slots into a list (because we can't pass a list of gradio components into a gradio command)
    # Number of slots isn't limted to 5, but currently hardcoded as such
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

def switchResult(type):

    if type == "Art":
        artResult = gr.Gallery.update(visible = True)
        videoResult = gr.Video.update(visible = False)
        return artResult, videoResult

    elif type == "Cinema":
        artResult = gr.Gallery.update(visible = False)
        videoResult = gr.Video.update(visible = True)
        return artResult, videoResult

### Classes

print("Creating classes...")

class dreamWorld:
    def __init__(
        self,
        prompt = "Soldiers fighting, close up, European city, steam punk, in the style of Jakub Rozalski, Caravaggio, volumetric lighting, sunset, cinematic lighting, highly detailed, masterpiece, fog, explosions ,depth of field",
        negativePrompt = "horses, ships, water, boat, modern, jpeg artifacts",
        width = 512,
        height = 512,
        scale = 7.5,
        steps = 32,
        seed = None,
        input_image = None,
        input_image_strength = 0.5,
        pytorchModel = None,
        batchSize = 1,
        saveSettings = True,
        jitCompile = False,
        animateFPS = 12,
        totalFrames = 24
    ):
        ## Let's create an object class that we can update later

        ## Set object variables
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.scale = scale
        self.steps = steps
        self.seed = seed
        self.input_image = input_image
        self.input_image_strength = input_image_strength
        self.pytorchModel = pytorchModel
        self.batchSize = batchSize
        self.saveSettings = saveSettings
        self.jitCompile = jitCompile
        self.animateFPS = animateFPS
        self.videoFPS = 24
        self.totalFrames = totalFrames
        self.generator = None

        ## Object variable corrections
        # Set seed if not given
        if self.seed is None or 0:
            self.seed = random.randint(0, 2 ** 31)

    def compileDreams(self):

        global programStarted
        global model
        global modelsLocation

        print(color.BLUE, color.BOLD,"\nStarting Stable Diffusion with Tensor flow and Apple Metal",color.END)
        programStarted = True

        # Which Mdoel to use? If the default tensorflow version is selected, then we'll download it!

        downloadWeights = False
        
        ## Object variable corrections
        # Set seed if not given
        if self.seed is None or 0:
            self.seed = random.randint(0, 2 ** 31)
        
        print("\nLoading Metal, connecting to GPU, and compiling Stable Diffusion")

        # Create generator with StableDiffusion class
        self.generator = StableDiffusion(
            img_height = int(self.height),
            img_width = int(self.width),
            jit_compile = self.jitCompile,
            download_weights = downloadWeights
        )

        # Are we using a pytroch model? If downloadWeights is fale, then yes we are!
        if self.pytorchModel != "Stable Diffusion 1.4":
            modelLocation = modelsLocation + self.pytorchModel
            self.generator.load_weights_from_pytorch_ckpt(modelLocation)
            model = self.pytorchModel
        
        print(color.GREEN,color.BOLD,"\nModel ready!",color.END)
    
    def create(
        self,
        type = "Art", # Which generation function to call. Art = still, Cinema = video
        prompt = "dinosaur riding a skateboard, cubism, textured, detailed",
        negativePrompt = "frame, framed",
        width = 512,
        height = 512,
        scale = 7.5,
        steps = 32,
        seed = None,
        inputImage = None,
        inputImageStrength = 0.5,
        pytorchModel = "StableDiffusion_V1p5.ckpt",
        batchSize = 1,
        saveSettings = True,
        projectName = "noProjectNameGiven",
        animateFPS = 12, # Starting from here down are video specific variables
        videoFPS = 24,
        totalFrames = 24,
        seedBehavior = "iter",
        saveVideo = True,
        angle = float("0"),
        zoom = float("1"),
        xTranslation = "0",
        yTranslation = "0",
        startingFrame = 0,
    ):
        
        # Update object variables that don't trigger a re-compile
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.scale = scale
        self.steps = steps
        self.seed = seed
        self.input_image = inputImage
        self.input_image_strength = inputImageStrength
        self.pytorchModel = pytorchModel
        self.batchSize = batchSize
        self.saveSettings = saveSettings
        # Video object variables that don't trigger
        self.animateFPS = animateFPS
        self.videoFPS = videoFPS
        self.totalFrames = int(totalFrames)

        # Update object variables that trigger a re-compile
        if width != self.width or height != self.height or batchSize != self.batchSize or pytorchModel != self.pytorchModel:
            # Load all of the re-compile variables
            self.width = int(width)
            self.height = int(height)
            self.pytorchModel = pytorchModel

            # Compile new model baesd on new parameters
            self.compileDreams()
        else:
            # Load all of the re-compile variables, but nothing has changed
            self.width = int(width)
            self.height = int(height)
            self.pytorchModel = pytorchModel

        # Global Variables
        global model

        # What to create?

        if type == "Art":
            # Create still image(s)
            result = self.generateArt()

            videoResult = None

            return result, videoResult
        elif type == "Cinema":
            # Create video
            result = None
            videoResult = self.generateCinema(
                projectName = projectName,
                seedBehavior = seedBehavior,
                angle = angle,
                zoom = zoom,
                xTranslation = xTranslation,
                yTranslation = yTranslation,
                saveVideo = saveVideo,
                startingFrame = int(startingFrame)
            )
            
            return result, videoResult

    def generateArt(self):

        # Before creation/generation, did we compile the model?
        if self.generator is None:
            self.compileDreams()

        print(color.PURPLE, "\nGenerating ",self.batchSize," image(s) of:", color.END)

        print(self.prompt)

        # Clear up tensorflow memory
        print("\n...cleaning memory...")
        tf.keras.backend.clear_session()
        gc.collect()

        # Use the generator function within the newly created class to generate an array that will become an image
        print("...getting to work...")
        imgs = self.generator.generate(
            prompt = self.prompt,
            negativePrompt = self.negativePrompt,
            num_steps = self.steps,
            unconditional_guidance_scale = self.scale,
            temperature = 1,
            batch_size = self.batchSize,
            seed = self.seed,
            input_image = self.input_image,
            input_image_strength = self.input_image_strength,
        )

        print(color.BOLD, color.GREEN, "\nFinished generating!")

        ### Create final image from the generated array ###

        # Generate PNG metadata for reference
        metaData = PngInfo()
        metaData.add_text('prompt', self.prompt)

        # Save settings
        if self.saveSettings is True:
            readWriteFile.writeToFile("creations/" + str(self.seed) + ".txt", [self.prompt, self.negativePrompt, self.width, self.height, self.scale, self.steps, self.seed, self.pytorchModel, self.batchSize, self.input_image_strength])

        # Multiple Image result:
        for img in imgs:
            print("Processing image!")
            imageFromBatch = Image.fromarray(img)
            imageFromBatch.save(creationLocation + str(int(self.seed)) + str(int(self.batchSize)) + ".png", pnginfo = metaData)
            print("Image saved!\n")
            self.batchSize = self.batchSize - 1

        print("Returning image!",color.END)
        return imgs
    
    def generateCinema(
        self,
        projectName = "noProjectNameGiven",
        animateFPS = 12,
        totalFrames = 24,
        seedBehavior = "Positive Iteration",
        angle = float("0"),
        zoom = float("1"),
        xTranslation = "0",
        yTranslation = "0",
        saveVideo = True,
        startingFrame = 0
    ):

        # Before creation/generation, did we compile the model?
        if self.generator is None:
            self.compileDreams()
        
        # Load in global variables
        global creationLocation

        print(color.PURPLE, "\nGenerating frames of:", color.END)

        print(self.prompt)

        # Local function variable creation
        seed = self.seed
        previousFrame = None
        currentFrame = self.input_image

        # Movement variabls
        angle = float(angle)
        zoom = float(zoom)

        if xTranslation is None:
            xTranslation = "0"
        
        if yTranslation is None:
            yTranslation = "0"

        originalTranslations = [xTranslation, yTranslation]
        xTranslation = videoUtil.generate_frames_translation(xTranslation, self.totalFrames)
        yTranslation = videoUtil.generate_frames_translation(yTranslation, self.totalFrames)

        # Load/create folder to save frames in
        path = f"content/{projectName}"
        if not os.path.exists(path): #If it doesn't exist, create folder
            os.makedirs(path)
        print("\nIn folder: ",path)

        # Save settings BEFORE running generation in case it crashes

        if self.saveSettings is True:
            readWriteFile.writeToFile(path + "/" + str(self.seed) + ".txt", [self.prompt, self.negativePrompt, self.width, self.height, self.scale, self.steps, self.seed, self.pytorchModel, self.batchSize, self.input_image_strength, self.animateFPS, self.videoFPS, self.totalFrames, seedBehavior, angle, zoom, originalTranslations[0], originalTranslations[1]])
        
        # Create frames
        for item in range(0, self.totalFrames):

            frameNumber = item + startingFrame

            print("\nGenerating Frame ",frameNumber)

            #if item > 0:
            #    currentFrame = videoUtil.maintain_colors(currentFrame, previousFrame)

            if startingFrame > 0 and item == 0:
                currentFrame = videoUtil.animateFrame2DWarp(
                    currentFrame,
                    angle = angle,
                    zoom = zoom,
                    xTranslation = xTranslation[item],
                    yTranslation = yTranslation[item],
                    width = self.width,
                    height = self.height
                )

            # Clear up tensorflow memory
            print("\n...cleaning memory...")
            tf.keras.backend.clear_session()
            gc.collect()

            # Use the generator function within the newly created class to generate an array that will become an image
            print("...getting to work...")
            
            previousFrame = currentFrame

            # Use the generator function within the newly created class to generate an array that will become an image
            frame = self.generator.generate(
                prompt = self.prompt,
                negativePrompt = self.negativePrompt,
                num_steps = self.steps,
                unconditional_guidance_scale = self.scale,
                temperature = 1,
                batch_size = self.batchSize,
                seed = seed,
                input_image = previousFrame,
                input_image_strength = self.input_image_strength
            )

            ## Save frame
            print(color.GREEN,"\nFrame generated. Saving to: ",path,color.END)
            # Generate metadata for saving in the png file
            metaData = PngInfo()
            metaData.add_text('seed:', str(int(seed)))
            metaData.add_text('prompt:', self.prompt)
            metaData.add_text('negative prompt:', self.negativePrompt)
            Image.fromarray(frame[0]).save(f"{path}/frame_{frameNumber:05}.png", format = "png", pnginfo = metaData)

            # Store frame array for next iteration
            currentFrame = videoUtil.animateFrame2DWarp(
                frame[0],
                angle = angle,
                zoom = zoom,
                xTranslation = xTranslation[item],
                yTranslation = yTranslation[item],
                width = self.width,
                height = self.height
            )
            
            #Memmory Clean Up
            frame = None
            previousFrame = None
            metaData = None

            # Update seed
            if seedBehavior == "Positive Iteration":
                seed = seed + 1
        
        print(color.GREEN,"\nCINEMA! Done",color.END)

        if saveVideo is True:
            finalVideo = self.deliverCinema(
                path, creationLocation, projectName
            )

            return finalVideo
    
    def deliverCinema(self, imagePath, videoPath, fileName):
        # Video creation

        imagePath = os.path.join(imagePath, "frame_%05d.png")
        videoPath = os.path.join(videoPath, f"{fileName}.mp4")

        videoUtil.constructFFmpegVideoCmd(self.animateFPS, self.videoFPS, imagePath, videoPath)

        return videoPath

print("...classes created. Starting program:")

dreamer = dreamWorld()

### Main Web UI Layout ###
# Define components outside of gradio's loop interface
# so they can be accessed regardless of child/parent position in the layout

## Main Tools

# Prompts
prompt = gr.Textbox(
    label = "Prompt - What should the AI create?"
)

negativePrompt = gr.Textbox(
    label = "Negative Prompt - What should the AI avoid when creating?"
)

# Creation Type

creationType = gr.Radio(
    choices = ["Art", "Cinema"],
    value = "Art",
    label = "Creation Type:"
)

# Start Button
startButton = gr.Button("Start")

startButton.style(
    full_width = True
)

## Basic Settings

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

# Batch Size

batchSizeSelect = gr.Slider(
    minimum = 1,
    maximum = int(batchMax),
    value = int(defaultBatchSize),
    step = 1,
    label = "Batch Size - How many results to make?"
)

## Elementary Settings

# Steps
steps = gr.Slider(
    minimum = 2,
    maximum = int(stepsMax),
    value = int(stepsMax) / 2,
    step = 1,
    label = "Steps - How many times the AI should sample - Higher numbers = better image"
)

# Scale
scale = gr.Slider(
    minimum = 2,
    maximum = int(scaleMax),
    value = 7.5,
    step = 0.1,
    label = "Guidance Scale - How closely should the AI follow the prompt - Higher number = follow more closely"
)

# Seed
seed = gr.Number(
    value = random.randint(0, 2 ** 31),
    label = "Seed  - Unique number for the image created",

)

## Advanced Settings

# Models/weights

modelsWeights = mf.findModels(modelsLocation, ".ckpt")

listOfModels = gr.Dropdown(
            choices = modelsWeights,
            label = "Model",
            value = defaultModel
        )

# Save user settings for prompt

saveSettings = gr.Checkbox(
    label = "Save settings used for prompt creation?",
    value = True
)

## Input Image

inputImage = gr.Image(
    label = "Input Image"
)

# Input Image Strength
inputImageStrength = gr.Slider(
    minimum = 0,
    maximum = 1,
    value = 0.5,
    step = 0.01,
    label = "Input Image Strength - 0 = Don't change the image, 1 = ignore image entirely"
)

# Prompt Engineering

starterPrompts = createPromptComponents(starterPrompt)

addPrompt = gr.Button("Add to prompt")

importPromptLocation = gr.File(
    label = "Import Prior Prompt and settings for prompt",
    type = "file"
)

importPromptButton = gr.Button("Import prompt")

## Video

# Project Name

projectName = gr.Textbox(
    value = "cinemaProject",
    label = "Name of the video - No spaces"
)

# FPS
# Animated
animatedFPS = gr.Dropdown(
    choices = [1,2,4,12,24,30,48,60],
    value = 12,
    label = "Animated Frames Per Second - 12 is standard animation"
)
# Final video
videoFPS = gr.Dropdown(
    choices = [24,30,60],
    value = 24,
    label = "Video Frames Per Second - 24 is standard cinema"
)

# Total frames
totalFrames = gr.Number(
    value = 48,
    label = "Total Frames",
)

# Starting frame

startingFrame = gr.Number(
    value = 0,
    label = "Starting Frame Number"
)

# Seed behavior
seedBehavior = gr.Dropdown(
    choices = ["Positive Iteration", "Negative Iteration", "Random Iteration", "Static Iteration"],
    value = "Positive Iteration",
    label = "Seed Behavior - How the seed changes from frame to frame"
)

# Save video
saveVideo = gr.Checkbox(
    label = "Save result as a video?",
    value = True
)

# Image Movement
# Angle
angle = gr.Slider(
    minimum = 0,
    maximum = 360,
    value = 0,
    step = 1,
    label = "Angle - Camera angle in degrees"
)

# Zoom
zoom = gr.Slider(
    minimum = 0.9,
    maximum = 1.1,
    value = 1,
    step = 0.01,
    label = "Zoom - Zoom in/out - Higher number zooms in"
)

# X Translation
xTranslate = gr.Textbox(
    label = "X Translation - Movement along x-axis",
    value = "-7"
)

# Y Translation
yTranslate = gr.Textbox(
    label = "Y Translation - Movement along y-axis",
    value = "-7"
)

## Tools

# Save and convert pytorch models

saveModelLocation = gr.Textbox(
    label = "Where to save the converted model",
    value = "models/"
)

saveModelTool = gr.Button("Save Model")

## Result(s)

# Gallery for still images
result = gr.Gallery(
    label = "Results",
)

result.style(
    grid = 2
)

resultVideo = gr.Video(
    label = "Result",
    visible = False
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
            # Creation type
            creationType.render()

            # Start Button
            startButton.render()

    # Image to Image
    with gr.Row():
        with gr.Column():
            with gr.Tab("Settings"):
                with gr.Row():
                    # Basic Settings
                    with gr.Column():
                        gr.Markdown("<center><b><u>Basic Settings</u></b></center>Necessary options")

                        # Width
                        width.render()

                        # Height
                        height.render()

                        # Batch Size
                        batchSizeSelect.render()
                    
                    # Elementary settings
                    with gr.Column():
                        gr.Markdown("<center><b><u>Elementary Settings</u></b></center>For more control")

                        # Steps
                        steps.render()

                        # Scale
                        scale.render()

                        # Seed
                        seed.render()

                        newSeed = gr.Button("New Seed")
            with gr.Tab("Advanced Settings"):
                # Model Selection
                listOfModels.render()

                # Save settings used for creation?
                saveSettings.render()

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
            
            with gr.Tab("Video"):
                
                gr.Markdown("<center><b>Video Settings</b></center>")

                projectName.render()

                animatedFPS.render()
                
                videoFPS.render()

                totalFrames.render()

                startingFrame.render()

                seedBehavior.render()

                saveVideo.render()

                gr.Markdown("<center><b>Movement Settings</b></center>")

                angle.render()

                zoom.render()

                xTranslate.render()

                yTranslate.render()

        with gr.Column():
            # Result
            with gr.Column():
                gr.Markdown("<center><span style = 'font-size: 24px'><b>Result</b></span></center>")
                
                result.render()

                resultVideo.render()

    ## Event actions

    # When start button is pressed
    startButton.click(
        fn = dreamer.create,
        inputs = [
            creationType,
            prompt,
            negativePrompt,
            width,
            height,
            scale,
            steps,
            seed,
            inputImage,
            inputImageStrength,
            listOfModels,
            batchSizeSelect,
            saveSettings,
            projectName,
            animatedFPS,
            videoFPS,
            totalFrames,
            seedBehavior,
            saveVideo,
            angle,
            zoom,
            xTranslate,
            yTranslate,
            startingFrame
        ],
        outputs = [result, resultVideo]
    )
    
    # When new seed is pressed
    newSeed.click(
        fn = randomSeed,
        inputs = None,
        outputs = seed,
    )

    # When add prompt is pressed
    addPrompt.click(
        fn = addToPrompt,
        inputs = [prompt, starterPrompts[0], starterPrompts[1], starterPrompts[2], starterPrompts[3], starterPrompts[4]],
        outputs = [prompt, starterPrompts[0], starterPrompts[1], starterPrompts[2], starterPrompts[3], starterPrompts[4]]
    )

    # When import button is pressed
    importPromptButton.click(
        fn = readWriteFile.readFromFile,
        inputs = importPromptLocation,
        outputs = [prompt, negativePrompt, width, height, scale, steps, seed, listOfModels, batchSizeSelect, inputImageStrength, animatedFPS, videoFPS, totalFrames, seedBehavior, angle, zoom, xTranslate, yTranslate]
    )

    # When creation type is selected
    creationType.change(
        fn = switchResult,
        inputs = creationType,
        outputs = [result, resultVideo]
    )

    ## Tools

print("\nLaunching Gradio\n")

demo.launch(
    inbrowser = True,
    show_error = True,
    share = True
)
