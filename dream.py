"""
Python modules
"""
### Console GUI
from rich import print, box
from rich.panel import Panel
from rich.text import Text

### Traceback
try:
    from rich.traceback import install
    install(show_locals = True)
except ImportError:
    print("Warning: Import error for Rich Traceback")
    pass    # no need to fail because of missing dev dependency

print(
    Panel(
        Text("MetalDiffusion", style = "bold grey89", justify = "center"),
        title = "Intel Mac",
        subtitle = "Apple Silicon",
        box = box.HEAVY,
        style = "white"
        )
    )
print("\n\nLoading program...")
### System modules
import os
import random
import argparse
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"

### Memory Management
import gc

print("\n...[bold]System[/bold] modules loaded...")

### Math modules
import numpy as np

### Import Stable Diffusion modules
from stableDiffusionTensorFlow.stableDiffusion import StableDiffusion
from stableDiffusionDiffusers.stableDiffusion import StableDiffusionDiffusers
print("...[bold]Stable Diffusion[/bold] modules loaded...")

### Machine Learning Modules

import tensorflow as tf
import torch as torch

print("...[bold]Machine Learning[/bold] modules loaded...")

### Computer Vision
import cv2

### Image saving after generation modules
from PIL import Image
from PIL.PngImagePlugin import PngInfo

print("...[bold]Image[/bold] modules loaded...")

### GUIModules
## WebUI
import gradio as gr
from GUI.gradioGUI import gradioGUIHandler, createLayout

print("...[bold]WebUI[/bold] module loaded...")

### Misc Modules
import utilities.modelWrangler as modelWrangler
import utilities.settingsControl as settingsControl
import utilities.readWriteFile as readWriteFile
import utilities.videoUtilities as videoUtil
import utilities.ImageTransformer as imageTransformer
import utilities.tensorFlowUtilities as tensorFlowUtilities
from utilities.consoleUtilities import color
import utilities.controlNetUtilities as controlNetUtilities
from utilities.depthMapping.run_pb import run as generateDepthMap
from utilities.tileSetter import setTiles

print("...[bold]Utilities[/bold] module loaded...")

print("...[green]all modules loaded![/green]")

"""
Command Line (CLI) Overrides
    This allows the user to override specific aspects of the Gradio implementation
"""

parser = argparse.ArgumentParser()

parser.add_argument(
    "--share",
    default = False,
    action = "store_true",
    help = "Share Gradio app publicly",
)

parser.add_argument(
    "--inBrowser",
    default = False,
    action = "store_true",
    help = "Automatically launch app in web browser",
)

CLIOverride = parser.parse_args()

"""
Functions
"""

def checkTime(start, end):
    totalMin = 0
    totalSec = 0
    totalHour = 0

    totalTime = end - start

    if totalTime > 60: #Convert to minutes
        totalMin = totalTime // 60
        totalSec = totalTime - (totalMin)*60
        if totalMin > 60: #Convert to hours
            totalHour = totalMin // 60
            totalMin = totalMin - (totalHour)*60
            print(totalHour,"hr ",totalMin,"min ",totalSec,"sec")
        else:
            print(totalMin,"min ",totalSec,"sec")
    else:
        totalSec = totalTime
        print(totalSec,"seconds")

    return totalTime

"""
Classes
"""

print("\n[bold]Creating classes...[/bold]")

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
        totalFrames = 24,
        VAE = "Original",
        textEmbedding = None,
        userSettings = None,
        availableWeights = None,
        allWeights = None,
        deviceChoices = []
    ):
        """
        dreamWorld - The Main Program Class

        This class handles all of the inputs, process, and outputs. It is independent of any GUI
        """
        ### Global Settings
        if userSettings == None:
            print("[yellow bold]No userPreferences.txt file found.[/yellow bold]\n[white]Creating one from fatory settings now...")
            # The factory settings are hard coded in the settingsControl.py file under createUserPreferences()
            userSettings = settingsControl.createUserPreferences(
                fileLocation = "userData/userPreferences.txt"
                )
        self.userSettings = userSettings
        ### Image Creation Variables
        ## Models
        self.pytorchModel = pytorchModel
        self.VAE = VAE
        self.textEmbedding = textEmbedding
        self.controlNetWeights = None
        self.availableWeights = availableWeights
        self.allWeights = allWeights
        self.LoRAs = []

        ## Creation Settings
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.steps = steps
        self.scale = scale
        self.seed = seed
        self.sampleMethod = None
        self.LoRAStrength = None

        ## Input Image
        self.input_image = input_image
        self.input_image_strength = input_image_strength

        ## Render Settings
        self.renderFramework = "Diffusers"
        self.legacy = True
        self.jitCompile = jitCompile
        self.optimizerMethod = "nadam" # For TensorFlow
        self.mixedPrecision = False
        self.device = "mps"
        self.deviceChoices = deviceChoices
        self.tokenMergingStrength = 0.0
        self.CLIPSkip = 0

        ## Video Settings
        self.animateFPS = animateFPS
        self.videoFPS = 24
        self.totalFrames = totalFrames

        ## Misc
        self.embeddingChoices = None
        self.saveSettings = saveSettings
        self.generator = None

        ## ControlNet
        self.controlNetProcess = None
        self.controlNetInput = None
        self.controlNetGuess = False
        self.controlNetStrength = 1
        self.controlNetCache = False
        self.controlNetSaveTiles = False

        ## Object variable corrections
        # Set seed if not given
        if self.seed is None or 0:
            self.seed = random.randint(0, 2 ** 31)

    def compileDreams(
            self,
            embeddingChoices = None,
            useControlNet = False
        ):

        # Time Keeping
        start = time.perf_counter()

        global model
        global ControlNetGradio

        if self.renderFramework == "Diffusers":
            print("\n[bold blue]Starting Stable Diffusion with :firecracker: Diffusers :firecracker:[/bold blue]\n")
        else:
            print("\n[bold blue]Starting Stable Diffusion with :ice: TensorFlow :ice:[/bold blue]\n")

        ## Memory Management
        self.generator = None
        gc.collect()
        
        ## Object variable corrections
        # Set seed if not given
        if self.seed is None or 0:
            self.seed = random.randint(0, 2 ** 31)
        
        ### Main Model Weights ###
        ## Expecting weights for TextEncoder, Diffusion Model, Encoder, and Decoder
        if self.pytorchModel is None:
            self.pytorchModel = self.userSettings["defaultModel"]
        modelKind = modelWrangler.findImportedModel(self.allWeights, self.pytorchModel)
        if modelKind != "huggingFace":
            modelLocation = self.userSettings["modelsLocation"] + modelKind + "/" + self.pytorchModel
        else:
            print("[yellow bold]NOTE:[/bold yellow] Model will be downloaded into cache from Hugging Face")
            modelLocation = self.pytorchModel
        
        ### VAE Weights
        ## Will replace Encoder and Decoder weights
        if self.VAE != "Original":
            VAELocation = self.userSettings["VAEModelsLocation"] + self.VAE
        else:
            VAELocation = "Original"
        
        ### Text Embedding Weights ###
        if embeddingChoices is not None:
            textEmbedding = []
            # print("Embedding Choices:",embeddingChoices)
            for choice in embeddingChoices:
                choice = choice.replace("<","")
                choice = choice.replace(">","")
                for embedding in self.textEmbedding:
                    if choice in embedding.lower():
                        print("Found <"+choice+"> as",embedding)
                        textEmbedding.append(embedding)
            if len(textEmbedding) == 0:
                print("Found no text embeddings")
                textEmbedding = None
            else:
                textEmbedding.insert(0,self.textEmbedding[0])
                #print("Passing these into model:\n",textEmbedding)
        else:
            textEmbedding = None

        ### ControlNet Weights ###
        if useControlNet is True:
            if self.controlNetWeights is not None:
                controlNetWeights = self.userSettings["ControlNetsLocation"] + self.controlNetWeights
            else:
                useControlNet = False
                controlNetWeights = None
        else:
            controlNetWeights = None
            useControlNet = False
        
        ### LoRA Weights ###
        if len(self.LoRAs) > 0:
            print("Using the following [bold]LoRA[/bold]'s:", self.LoRAs)
            self.LoRAs.insert(0,self.userSettings["LoRAsLocation"])

        # Create generator with StableDiffusion class

        if self.renderFramework == "Diffusers":
            self.generator = StableDiffusionDiffusers(
                imageHeight = int(self.height),
                imageWidth = int(self.width),
                jit_compile = self.jitCompile,
                weights = modelLocation,
                VAE = VAELocation,
                mixedPrecision = self.mixedPrecision,
                textEmbeddings = textEmbedding,
                controlNet = [useControlNet, controlNetWeights],
                device = self.device,
                LoRAs = self.LoRAs,
                tokenMergingStrength = self.tokenMergingStrength,
                CLIPSkip = self.CLIPSkip
            )
        else:
            # TensorFlow
            self.generator = StableDiffusion(
                imageHeight = int(self.height),
                imageWidth = int(self.width),
                jit_compile = self.jitCompile,
                weights = modelLocation,
                legacy = self.legacy,
                VAE = VAELocation,
                mixedPrecision = self.mixedPrecision,
                textEmbeddings = textEmbedding,
                controlNet = [useControlNet, controlNetWeights],
                device = self.device
            )
        
        print("[green bold]\nModels ready![/green bold]")

        # Time keeping
        end = time.perf_counter()
        checkTime(start, end)
    
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
        xTranslation = 0.0, yTranslation = 0.0, zTranslation = 0.0,
        xRotation = 0.0, yRotation = 0.0, zRotation = 0.0,
        focalLength = 200.0,
        startingFrame = 0,
        legacy = True,
        VAE = "Original",
        embeddingChoices = None,
        mixedPrecision = True,
        sampleMethod = None,
        optimizerMethod = "nadam",
        deviceOption = '/gpu:0',
        useControlNet = False,
        controlNetWeights = None,
        controlNetProcess = None,
        controlNetInput = None,
        controlNetGuess = False,
        controlNetStrength = 1,
        controlNetCache = False,
        controlNetLowThreshold = 100,
        controlNetHighThreshold = 200,
        controlNetTileUpscale = None,
        controlNetUpscaleMethod = None,
        controlNetSaveTiles = False,
        vPrediction = False,
        reuseInputImage = False,
        reuseControlNetInput = False,
        renderFramework = "Diffusers",
        LoRAChoices = None,
        LoRAStrength = 0.5,
        tokenMergingStrength = 50,
        CLIPSkip = 0
    ):
        
        # Import global variables
        #global userSettings
        global model

        ### Update object variables that don't trigger a re-compile/build, but do influence it
        ## Image Creation
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.scale = scale
        self.steps = steps
        self.seed = seed
        self.batchSize = batchSize
        self.sampleMethod = sampleMethod

        ## Input Image
        self.input_image = inputImage
        self.input_image_strength = inputImageStrength

        ## Video
        self.animateFPS = animateFPS
        self.videoFPS = videoFPS
        self.totalFrames = int(totalFrames)
        xyzTranslation = [float(xTranslation), float(yTranslation), float(zTranslation)]
        xyzRotation = [float(xRotation), float(yRotation), float(zRotation)]

        ## Misc
        self.saveSettings = saveSettings
        self.controlNetSaveTiles = controlNetSaveTiles

        # Modes
        self.legacy = legacy
        if mixedPrecision is True:
            self.mixedPrecision = mixedPrecision
            if self.generator is not None:
                self.generator.changePolicy("mixed_float16")
        else:
            self.mixedPrecision = mixedPrecision
            if self.generator is not None:
                self.generator.changePolicy("float32")

        # Device Selection
        for device in self.deviceChoices:
            if device['name'] == deviceOption:
                selectedDevice = device['TensorFlow'].name[-1]
                if "CPU" in device['TensorFlow'].name:
                    print("[cyan]\nUsing CPU to render:\n[/cyan]",device['name'])
                    selectedDevice = "/device:CPU:" + selectedDevice
                    if renderFramework == "Diffusers":
                        self.device = "cpu"
                    else:
                        self.device = selectedDevice
                elif "GPU" in device['TensorFlow'].name:
                    print("[cyan]\nUsing GPU to render:\n[/cyan]",device['name'])
                    selectedDevice = "/GPU:" + selectedDevice
                    if renderFramework == "Diffusers":
                        self.device = "mps"
                    else:
                        self.device = selectedDevice
        
        # Text Embeddings
        if len(embeddingChoices) ==  0:
            # No given embeddings means ignoring text embeddings
            embeddingChoices = None
        
        # ControlNet Bypass, cancels out any user inputs if user explicity says "Don't use ControlNet"
        if useControlNet is False:
            controlNetWeights = None
            controlNetProcess = None
            controlNetInput = None
            controlNetCache = None
        
        # LoRA
        self.LoRAStrength = LoRAStrength

        # Prep Token Merging Strength
        tokenMergingStrength = float(tokenMergingStrength) / 100

        # Prep CLIP Skip
        CLIPSkip = int(CLIPSkip)

        ### Critical Changes that require a re-compile

        if width != self.width or height != self.height or embeddingChoices != self.embeddingChoices or controlNetWeights != self.controlNetWeights or renderFramework != self.renderFramework or LoRAChoices != self.LoRAs or tokenMergingStrength != self.tokenMergingStrength or CLIPSkip != self.CLIPSkip:
            print("[yellow]\n[/][bold][yellow]Critical changes made for creation[/yellow][/bold], compiling/building new model")
            print("\nNew inputs:",renderFramework,"\n","Width:",width,"Height:",height,"Batch Size:",batchSize,"\n  Embeddings:",embeddingChoices,"\n   ControlNet:",controlNetWeights,"\n    LoRAs:",LoRAChoices, "\n     Token Merging Strength:", tokenMergingStrength, "\n      CLIP Skip:", CLIPSkip)
            print("\nOld inputs:",self.renderFramework,"\n","Width:",self.width,"Height:",self.height,"Batch Size:",self.batchSize,"\n  Embeddings:",self.embeddingChoices,"\n   ControlNet:",self.controlNetWeights,"\n    LoRAs:",self.LoRAs, "\n     Token Merging Strength:", self.tokenMergingStrength, "\n      CLIP Skip:", self.CLIPSkip)
            # Basic Variables
            self.width = int(width)
            self.height = int(height)
            self.pytorchModel = pytorchModel
            self.VAE = VAE
            self.renderFramework = renderFramework
            
            ## Text Embeddings
            self.embeddingChoices = embeddingChoices

            ### ControlNet
            self.controlNetWeights = controlNetWeights

            ## LoRAs
            self.LoRAs = LoRAChoices

            ### Token Merging
            self.tokenMergingStrength = tokenMergingStrength

            ### CLIP Skip
            self.CLIPSkip = CLIPSkip

            # Compile new model baesd on new parameters
            self.compileDreams(embeddingChoices = embeddingChoices, useControlNet = useControlNet)
        else:
            # Basic Variables
            self.width = int(width)
            self.height = int(height)

            ## Text Embeddings
            self.embeddingChoices = embeddingChoices

            ### ControlNet
            self.controlNetWeights = controlNetWeights

            ## LoRAS
            self.LoRAs = LoRAChoices

            ### Token Merging
            self.tokenMergingStrength = tokenMergingStrength
        
        ### Regular Changes that do not require re-compile ever

        ## Weights

        if pytorchModel != self.pytorchModel:
            print("\n[blue]New model weights selected!\nApplying weights from:\n[/blue]",pytorchModel)

            # Main Model Weights
            self.pytorchModel = pytorchModel
            modelKind = modelWrangler.findImportedModel(self.allWeights, self.pytorchModel)
            modelLocation = self.userSettings["modelsLocation"] + modelKind + "/" + self.pytorchModel

            # VAE Weights
            self.VAE = VAE
            if self.VAE != "Original":
                VAELocation = self.userSettings["VAEModelsLocation"] + self.VAE
            else:
                VAELocation = "Original"
            
            # Update weights

            if self.generator is not None:
                self.generator.setWeights(modelLocation, VAELocation)
            else:
                self.compileDreams(embeddingChoices = embeddingChoices, useControlNet = useControlNet)
        else:
            ("[blue]Using model weights:\n[/blue]",pytorchModel,color.END)
            self.pytorchModel = pytorchModel
            self.VAE = VAE

        if optimizerMethod != self.optimizerMethod:
            print("New optimizer!")
            #self.generator.compileModels(optimizerMethod, True)
            self.optimizerMethod = optimizerMethod
        
        ## ControlNet
        if useControlNet is True:
            # Pre-Process Option
            if controlNetProcess == "None":
                print("User selected no processing for controlnet")
                controlNetProcess = "BYPASS"
            self.controlNetProcess = controlNetProcess
            
            # ControlNet Input Image
            if controlNetInput is not None:
                controlNetInput = controlNetUtilities.preProcessControlNetImage(
                    image = controlNetInput,
                    processingOption = self.controlNetProcess,
                    imageSize = [self.width, self.height],
                    cannyOptions = [controlNetLowThreshold, controlNetHighThreshold],
                    tileScale = int(controlNetTileUpscale),
                    upscaleMethod = controlNetUpscaleMethod
                )
            
            # Checking if input has changed for cache
            if self.controlNetInput is not None:
                # this means we've already done one generation with a controlNet input
                if np.array_equal(self.controlNetInput, controlNetInput):
                    print("ControlNet Inputs match!")
                else:
                    print("Different ControlNet inputs!")
            
            self.controlNetInput = controlNetInput

            if self.renderFramework == "Diffusers":
                if len(self.controlNetInput) < 1:
                    self.controlNetInput = Image.fromarray(self.controlNetInput[0])
                    self.controlNetInput = [self.controlNetInput]
                else:
                    for index, tile in enumerate(self.controlNetInput):
                        self.controlNetInput[index] = Image.fromarray(tile)
            
            # Strength of ControlNet
            self.controlNetGuess = controlNetGuess
            if isinstance(controlNetStrength, list):
                self.controlNetStrength = float(controlNetStrength[0])
            else:
                self.controlNetStrength = float(controlNetStrength)

            if self.controlNetGuess is True:
                if self.renderFramework != "Diffusers":
                    self.controlNetStrength = [self.controlNetStrength * (0.825 ** float(12 - i)) for i in range(13)]
                else:
                    self.controlNetStrength = [self.controlNetStrength, 1337]
                    #We'll pass on a list with length of two to the Diffusers framework, indicating Guess Mode selected
            else:
                if self.renderFramework != "Diffusers":
                    self.controlNetStrength = [self.controlNetStrength] * 13

            # Use Cache?
            self.controlNetCache = controlNetCache
        else:
            self.controlNetWeights = controlNetWeights
            self.controlNetProcess = controlNetProcess
            self.controlNetInput = controlNetInput
            self.controlNetCache = controlNetCache
            if self.renderFramework != "Diffusers":
                self.controlNetStrength = [1] * 13
            else:
                self.controlNetStrength = 1

        ### What to create? ###

        if type == "Art":
            # Create still image(s)
            result = self.generateArt(sampleMethod = self.sampleMethod, vPrediction = vPrediction)

            videoResult = None

            return result, videoResult
        elif type == "Cinema":
            # Create video
            result = None

            videoResult = self.generateCinema(
                projectName = projectName,
                seedBehavior = seedBehavior,
                xyzTranslation = xyzTranslation,
                xyzRotation = xyzRotation,
                focalLength = float(focalLength),
                reuseInputImage = reuseInputImage,
                saveVideo = saveVideo,
                startingFrame = int(startingFrame),
                sampleMethod = self.sampleMethod,
                vPrediction = vPrediction,
                reuseControlNetInput = reuseControlNetInput
            )
            
            return result, videoResult

    def generateArt(
            self,
            sampleMethod = None,
            vPrediction = False
        ):
        # Global variables
        global userSettings
        
        # Time Keeping
        start = time.perf_counter()

        # Save settings
        if self.saveSettings is True:
            readWriteFile.writeToFile("creations/" + str(self.seed) + ".txt", [self.prompt, self.negativePrompt, self.width, self.height, self.scale, self.steps, self.seed, self.pytorchModel, self.batchSize, self.input_image_strength, self.animateFPS, self.videoFPS, self.totalFrames, "Static", "0", "1", "0", "0", self.controlNetWeights, self.controlNetStrength])

        # Before creation/generation, do we have a compiled/built model?
        if self.generator is None:
            self.compileDreams()

        print("[purple]\nGenerating ",self.batchSize,"[purple] image(s) of:[/purple]")

        print(self.prompt)

        if self.controlNetInput is not None and len(self.controlNetInput) > 1:
            print("\n[gold]Tile mode activated.[/gold] Rendering:",str( len(self.controlNetInput) ),"tiles")
            # Create variables
            imgs = []
            resultingTiles = []
            print("Number of tiles:",len(self.controlNetInput))
            tileProgress = 1

            for tile in self.controlNetInput:
                print("[blue]\nProcessing Tile Number[/blue]", tileProgress)
                if self.renderFramework == "TensorFlow":
                    tileInputImage = tile
                    tileControlNetInput = tf.constant(tile.copy(), dtype = tf.float32) / 255.0
                    tileControlNetInput = [tileControlNetInput]
                else:
                    tileInputImage = tile
                    tileControlNetInput = tile
                # Use the generator function within the newly created class to generate an array that will become an image
                imgs = self.generator.generate(
                    prompt = self.prompt,
                    negativePrompt = self.negativePrompt,
                    num_steps = self.steps,
                    unconditional_guidance_scale = self.scale,
                    temperature = 1,
                    batch_size = self.batchSize,
                    seed = self.seed,
                    input_image = tileInputImage,
                    input_image_strength = self.input_image_strength,
                    sampler = sampleMethod,
                    controlNetStrength = self.controlNetStrength,
                    controlNetImage = tileControlNetInput,
                    controlNetCache = self.controlNetCache,
                    vPrediction = vPrediction,
                    LoRAStrength = self.LoRAStrength
                )

                print("\n[bold green]Tile Done![/bold green]")

                ### Create final image from the generated array ###

                # Generate PNG metadata for reference
                metaData = self.createMetadata()

                # Multiple Image result:
                for img in imgs:
                    print("Processing tile...")
                    if self.controlNetSaveTiles == True:
                        print("...saving tiles...")
                        if isinstance(img, np.ndarray):
                            imageFromBatch = Image.fromarray(img)
                        else:
                            imageFromBatch = img
                        imageFromBatch.save(self.userSettings["creationLocation"] + str(int(self.seed)) + "_TILE00" + str(int(tileProgress)) + ".png", pnginfo = metaData)
                        print("...tile saved...")
                    if isinstance(img,np.ndarray) is False:
                        img = np.array(img)
                    resultingTiles.append(img)
                    print("...tile processed and added to collection!")

                # Update tile progress
                tileProgress += 1
            
            # Combine all tiles togeter
            print("Tiles done! Setting tiles now...")
            finalImage = setTiles(resultingTiles)
            finalImage.save(self.userSettings["creationLocation"] + str(int(self.seed)) + "_FINAL.png", pnginfo = metaData)
            print("[green bold]Completed![/green bold] [green]Returning final image[/green]")

            # Time keeping
            end = time.perf_counter()
            checkTime(start, end)
            return [finalImage]
        else:
            if self.renderFramework == "Diffusers":
                if self.controlNetInput is not None:
                    self.controlNetInput = self.controlNetInput[0]
            # Use the generator function within the newly created class to generate an array that will become an image
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
                sampler = sampleMethod,
                controlNetStrength = self.controlNetStrength,
                controlNetImage = self.controlNetInput,
                controlNetCache = self.controlNetCache,
                vPrediction = vPrediction,
                LoRAStrength = self.LoRAStrength
            )

            print("[bold green]\nFinished generating![/bold green]")

            ### Create final image from the generated array ###

            # Generate PNG metadata for reference
            metaData = self.createMetadata()

            # Multiple Image result:
            imageCount = 0
            print("Processing image(s)...")
            for img in imgs:
                if isinstance(img, np.ndarray):
                    imageFromBatch = Image.fromarray(img)
                else:
                    imageFromBatch = img
                imageFromBatch.save(self.userSettings["creationLocation"] + str(int(self.seed)) + str(int(self.batchSize)) + ".png", pnginfo = metaData)
                print("...image(s) saved!\n\a\a\a")
                self.batchSize = self.batchSize - 1
                #if isinstance(img, np.ndarray) is False:
                    #imgs[imageCount] = np.array(img)

            print("[green]Returning image![/green]")

            # Time keeping
            end = time.perf_counter()
            checkTime(start, end)

            return imgs
    
    def generateCinema(
        self,
        projectName = "noProjectNameGiven",
        seedBehavior = "Positive Iteration",
        xyzTranslation = [0.0, 0.0, 200.0],
        xyzRotation = [90.0, 90.0, 90.0],
        focalLength = 200.0,
        reuseInputImage = False,
        saveVideo = True,
        startingFrame = 0,
        sampleMethod = None,
        vPrediction = False,
        reuseControlNetInput = False
    ):

        # Before creation/generation, did we compile the model?
        if self.generator is None:
            self.compileDreams()
        
        # Load in global variables
        #global userSettings

        print("[purple]\nGenerating frames of:[/purple]")

        print(self.prompt)

        # Local variables
        seed = self.seed
        previousFrame = self.input_image
        currentInputFrame = None
        renderTime = 0

        # Load/create folder to save frames in
        path = f"content/{projectName}"
        if not os.path.exists(path): #If it doesn't exist, create folder
            os.makedirs(path)
        print("\nIn folder: ",path)

        # Movement variables
        #angle = float(angle)
        #zoom = float(zoom)
        
        print("...giving camera direction...")
        originalTranslations = xyzTranslation.copy()
        originalRotations = xyzRotation.copy()

        # Save settings BEFORE running generation in case it crashes

        if self.saveSettings is True:
            readWriteFile.writeToFile(path + "/" + str(self.seed) + ".txt", [self.prompt, self.negativePrompt, self.width, self.height, self.scale, self.steps, self.seed, self.pytorchModel, self.batchSize, self.input_image_strength, self.animateFPS, self.videoFPS, self.totalFrames, seedBehavior, originalTranslations[0], originalTranslations[1], self.controlNetWeights, self.controlNetStrength])
        
        # Create frames
        for item in range(0, self.totalFrames): # Minus 1 from total frames because we're starting at 0 instead of 1 when counting up. User asks for 24 frames, computer counts from 0 to 23

            # Time Keeping
            start = time.perf_counter()

            # Update frame number
            # If starting frame is given, then we're also adding every iteration to the number
            frameNumber = item + startingFrame

            print("\nGenerating Frame ",frameNumber)

            # Continue camera movement from prior frame if starting frame was given
            if startingFrame > 0 and item == 0:
                print("\n...continuing camera movement...")
                previousFrame = imageTransformer.rotateImage(
                    previousFrame,
                    xyzTranslation[0],
                    xyzTranslation[1],
                    xyzTranslation[2],
                    xyzRotation[0],
                    xyzRotation[1],
                    xyzRotation[2],
                    focalLength
                )
            
            if reuseControlNetInput is True:
                print("\nReusing Initial ControlNet Input Image")
            else:
                if previousFrame is not None and frameNumber > 0:
                    if self.controlNetWeights != None:
                        self.controlNetInput = controlNetUtilities.preProcessControlNetImage(previousFrame, self.controlNetProcess, imageSize = [self.width, self.height])
            
            # Color management
            if currentInputFrame is not None:
                print("...maintaning [red]c[/red][yellow]o[/yellow][green]l[/green][blue]o[/blue][magenta]r[/magenta][white]s[/white]...")
                if isinstance(currentInputFrame, np.ndarray) is False:
                    print("...converting currentInputFrame from 'PIL' to 'numpy' for color correction...")
                    currentInputFrame = np.array(currentInputFrame)
                previousFrame = videoUtil.maintainColors(previousFrame, currentInputFrame)
            
            # If user chooses to only use the initial input image
            if reuseInputImage is True:
                print("\n...reusing [bold]Initial Input Image[/bold]...")
                previousFrame = self.input_image
            
            # Update previous frame variable for use in the generation of this frame
            if self.renderFramework == "Diffusers":
                if frameNumber == 0:
                    # The tweaked Diffusers Render Framework is expecting an image with BGR instead of RGB and will convert it to RGB
                    # Since this is the first frame, we can bypass that conversion by passing a PIL image instead of an nparray(which indicates cv2 was used and results in BGR)
                    if previousFrame is not None:
                        print("...converting initial image 'numpy' to 'PIL' for inference...")
                        currentInputFrame = Image.fromarray(previousFrame).convert("RGB")
                    else:
                        currentInputFrame = previousFrame
                else:
                    # The tweaked Diffusers Render Framework is expecting an image with BGR instead of RGB and will convert it to RGB
                    # previousFrame = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2RGB)
                    print("...converting previous frame 'numpy' to 'PIL' for inference...")
                    previousFrame = Image.fromarray(previousFrame).convert("RGB")
                    currentInputFrame = previousFrame
                
                # Debug
                currentInputFrame.save(f"debug/frameAfterWarpBeforeInference_{frameNumber:05}.png", format = "png")
            else:
                currentInputFrame = previousFrame

            ## Create frame
            # frame variable calls the generator to generate an image
            frame = self.generator.generate(
                prompt = self.prompt,
                negativePrompt = self.negativePrompt,
                num_steps = self.steps,
                unconditional_guidance_scale = self.scale,
                temperature = 1,
                batch_size = self.batchSize,
                seed = seed,
                input_image = currentInputFrame,
                input_image_strength = self.input_image_strength,
                sampler = sampleMethod,
                controlNetStrength = self.controlNetStrength,
                controlNetImage = self.controlNetInput,
                controlNetCache = self.controlNetCache,
                vPrediction = vPrediction,
                LoRAStrength = self.LoRAStrength
            )

            ## Save frame
            print("[green]\nFrame generated. Saving to: \a[/green]",path)

            # Generate metadata for saving in the png file
            metaData = self.createMetadata()
            
            if isinstance(frame[0], np.ndarray) is False:
                frame = np.array(frame[0])
            else:
                frame = frame[0]
            savedImage = Image.fromarray(frame)
            savedImage.save(f"{path}/frame_{frameNumber:05}.png", format = "png", pnginfo = metaData)

            # Store frame array for next iteration
            print("...applying camera movement for next frame...")

            previousFrame = imageTransformer.rotateImage(
                frame,
                xyzTranslation[0],
                xyzTranslation[1],
                xyzTranslation[2],
                xyzRotation[0],
                xyzRotation[1],
                xyzRotation[2],
                focalLength
            )
            
            # Memmory Clean Up
            frame = None
            metaData = None
            savedImage = None
            gc.collect()

            # Update seed
            seed = videoUtil.nextSeed(seedBehavior = seedBehavior, seed = seed)
            
            # Time keeping
            end = time.perf_counter()
            renderTime = renderTime + checkTime(start, end)
        
        # Finished message and time keeping
        print("[green bold]\nCINEMA! Created in:\a[/green bold]")
        checkTime(0, renderTime)
        print("Per frame:")
        checkTime(0, renderTime/(self.totalFrames))

        ## Video compiling
        if saveVideo is True:
            finalVideo = self.deliverCinema(
                path, self.userSettings["creationLocation"], projectName
            )

            return finalVideo
    
    def deliverCinema(self, imagePath, videoPath, fileName):
        # Video creation

        imagePath = os.path.join(imagePath, "frame_%05d.png")
        videoPath = os.path.join(videoPath, f"{fileName}.mp4")

        videoUtil.constructFFmpegVideoCmd(self.animateFPS, self.videoFPS, imagePath, videoPath)

        return videoPath
    
    def createMetadata(self):
        # Metadata to be stored in the image file
        metaData = PngInfo()
        metaData.add_text('prompt', self.prompt)
        metaData.add_text('negative prompt', self.negativePrompt)
        metaData.add_text('seed', str(int(self.seed)))
        metaData.add_text('CFG scale', str(self.scale))
        metaData.add_text('steps', str(int(self.steps)))
        metaData.add_text('input image strength', str(self.input_image_strength))
        if isinstance(self.controlNetStrength, list):
            controlNetStrength = int(self.controlNetStrength[0])
        else:
            controlNetStrength = self.controlNetStrength
        metaData.add_text('controlNet strength',str(controlNetStrength))
        metaData.add_text('model', self.pytorchModel)
        metaData.add_text('batch size',str(self.batchSize))
        metaData.add_text('sampler', str(self.sampleMethod))
        metaData.add_text('render framework', str(self.renderFramework))
        metaData.add_text('program','MetalDiffusion')

        return metaData
    
    def switchWeights(self, type):
        if type == "Diffusers":
            self.availableWeights = self.allWeights["safetensors"].copy()
            self.availableWeights.extend(self.allWeights["diffusers"].copy())
            self.availableWeights.extend(self.allWeights["huggingFace"].copy())
            self.availableWeights.extend(self.allWeights["ckpt"].copy())
        elif type == "TensorFlow":
            self.availableWeights = self.allWeights["safetensors"].copy()
            self.availableWeights.extend(self.allWeights["tensorflow"].copy())
            self.availableWeights.extend(self.allWeights["ckpt"].copy())

        return self.availableWeights

print("...done!")

"""
Global variables
"""

### Global Variables
print("\n[bold]Creating global variables...[/bold]")
endProgramVariable = False
model = None
dreamer = None

# Try loading custom settings from user file, otherwise continue with factory settings
print("...loading user preferences...")
userSettings = settingsControl.loadSettings("userData/userPreferences.txt")
if userSettings is False: # This means loadSettings() couldn't find the file. Time to create one
    print("...[yellow]no user preferences found[/yellow]...")
    userSettings = settingsControl.createUserPreferences(
        fileLocation = "userData/userPreferences.txt"
    )

## Prompt Settings
try:
    starterPrompt = settingsControl.loadSettings("userData/promptGenerator.txt", 1)
except Exception as e:
    print(e)
    starterPrompt = []

## Devices

deviceChoices = tensorFlowUtilities.listDevices()

print("...[green]global variables created![/green]")

"""
Load all models and weights
"""

print("\nSearching for [blue]diffusion[/blue] models/weights...\n")
allWeights, currentWeights, VAEWeights, embeddingWeights, embeddingNames, controlNetWeights, LoRAs = modelWrangler.findAllWeights(userSettings = userSettings)

print("[bold green]\nStarting program:\a[/bold green]")

"""
Main Class
    This is the object we'll be referencing in the Web UI
"""
dreamer = dreamWorld(
    textEmbedding = embeddingWeights,
    userSettings = userSettings,
    availableWeights = currentWeights,
    allWeights = allWeights,
    deviceChoices = deviceChoices
)

"""
GUI
"""

gradioGUI = gradioGUIHandler(
    dreamer = dreamer,
    currentWeights = currentWeights,
    embeddingNames = embeddingNames,
    LoRAs = LoRAs,
    controlNetWeights = controlNetWeights,
    VAEWeights = VAEWeights,
    starterPrompt = starterPrompt,
    deviceChoices = deviceChoices
)

finalGradioGUI = createLayout(
    gradioGUI = gradioGUI
)

"""
Final Steps
"""

print("[blue]\nLaunching Gradio:\n[/blue]")

finalGradioGUI.launch(
    inbrowser = CLIOverride.inBrowser,
    show_error = True,
    share = CLIOverride.share
)