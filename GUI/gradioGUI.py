"""
Gradio GUI
-----------

Hello! I am the script written by AJ Young that creates the GUI for MetalDiffusion via Gradio.

More info about Gradio:

"""
### System modules
import os
import random
import signal

### Console GUI
from rich import print, box
from rich.panel import Panel
from rich.text import Text

## WebUI
import gradio as gr

### Utilities
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

### Math Modules
import numpy as np

"""
Main Class
"""

class gradioGUIHandler:
    def __init__(
            self,
            dreamer = None,
            currentWeights = [],
            embeddingNames = [],
            LoRAs = [],
            controlNetWeights = [],
            VAEWeights = [],
            starterPrompt = [],
            deviceChoices = []
    ):
        if dreamer == None:
            print("[red bold]FATAL ERROR[/red bold][yellow]\nNo dreamer class was given. Exiting Gradio GUI now[/yellow]")
        
        ### What is our main class to reference?
        self.dreamer = dreamer

        """
        Main Web Components
            Define components outside of gradio's loop interface
            so they can be accessed regardless of child/parent position in the layout
        """

        self.render = Render(
            dreamer = dreamer
        )

        self.imageSettings = ImageSettings(
            dreamer = dreamer,
            currentWeights = currentWeights,
            embeddingNames = embeddingNames,
            LoRAs = LoRAs
        )        

        self.controlNet = ControlNet(
            dreamer = dreamer,
            controlNetWeights = controlNetWeights
        )

        self.advancedSettings = AdvancedSettings(
            dreamer = dreamer,
            VAEWeights = VAEWeights,
            deviceChoices = deviceChoices
        )

        self.Import = Import(
            dreamer = dreamer
        )

        self.animation = Animation()

        """
        Tools
        """

        self.depthMapping = DepthMapping()

        self.promptEngineering = PromptEngineering(
            embeddingNames = embeddingNames,
            starterPrompt = starterPrompt
        )

        self.modelConversion = ModelConversion()

        self.videoTools = VideoTools(
            dreamer = dreamer
        )

        self.modelAnalysis = PyTorchModelAnalysis()

        """
        Results
        """

        # Gallery for still images
        self.result = gr.Gallery(
            label = "Results",
            height = 896
        ).style(height = 896)

        self.result.style(
            grid = 2
        )

        self.resultVideo = gr.Video(
            label = "Result",
            visible = False
        ).style(height = 896)

        self.previewResult = gr.Image(
            label = "Preview",
            visible = True
        ).style(height = 896)

        ## End Program

        self.endProgramButton = gr.Button(
            "Close Program"
        )
    
    def addToPrompt(
            self,
            originalPrompt,
            embeddings,
            slotA,
            slotB,
            slotC,
            slotD,
            slotE
        ):
        # Combine slots into a list (because we can't pass a list of gradio components into a gradio command)
        # Number of slots isn't limted to 5, but currently hardcoded as such
        additionList = [embeddings, slotA, slotB, slotC, slotD, slotE]
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

    def endProgram(self):
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)
    
    def previewImageTransform(
            self,
            inputImage,
            xTranslation,
            yTranslation,
            zTranslation,
            xRotation,
            yRotation,
            zRotation,
            focalLength
        ):

        if isinstance(inputImage, np.ndarray) is False:
            if inputImage == None:
                print("No input image given!\nGo to the 'Input Image Tab' and add an image.")
                return None
        print("\nPreviewing image transformation with:")

        previewImage = imageTransformer.rotateImage(
            inputImage,
            float(xTranslation),
            float(yTranslation),
            float(zTranslation),
            float(xRotation),
            float(yRotation),
            float(zRotation),
            float(focalLength)
        )

        print("[bold green]Done![/bold green] [green]Returning image[/green]...")

        return previewImage

    def randomSeed(self):
        newSeed = random.randint(0, 2 ** 31)
        return newSeed

    def saveModel(
            self,
            saveModelName = None,
            listOfModels = None,
            legacyVersion = None,
            saveModelType = None
    ):
        modelWrangler.saveModel(
            saveModelName,
            pytorchModel = listOfModels,
            legacyMode = legacyVersion,
            typeOfModel = saveModelType,
            dreamer = self.dreamer,
            currentWeights = self.dreamer.allWeights,
            userSettings = self.dreamer.userSettings
        )
    
    def switchControlNetOptions(self, type):

        if type == "None":
            # Canny
            controlNetThresholdLow = gr.Slider.update(visible = False)
            controlNetThresholdHigh = gr.Slider.update(visible = False)
            # Tile
            controlNetTileSize = gr.Slider.update(visible = False)
            controlNetUpscaleMethod = gr.Slider.update(visible = False)
            controlNetTileSaveTiles = gr.Checkbox.update(visible = False)
        elif type == "Canny":
            # Canny
            controlNetThresholdLow = gr.Slider.update(visible = True)
            controlNetThresholdHigh = gr.Slider.update(visible = True)
            # Tile
            controlNetTileSize = gr.Slider.update(visible = False)
            controlNetUpscaleMethod = gr.Slider.update(visible = False)
            controlNetTileSaveTiles = gr.Checkbox.update(visible = False)
        elif type == "HED":
            # Canny
            controlNetThresholdLow = gr.Slider.update(visible = False)
            controlNetThresholdHigh = gr.Slider.update(visible = False)
            # Tile
            controlNetTileSize = gr.Slider.update(visible = False)
            controlNetUpscaleMethod = gr.Slider.update(visible = False)
            controlNetTileSaveTiles = gr.Checkbox.update(visible = False)
        elif type == "Tile":
            # Canny
            controlNetThresholdLow = gr.Slider.update(visible = False)
            controlNetThresholdHigh = gr.Slider.update(visible = False)
            # Tile
            controlNetTileSize = gr.Slider.update(visible = True)
            controlNetUpscaleMethod = gr.Slider.update(visible = True)
            controlNetTileSaveTiles = gr.Checkbox.update(visible = True)
        
        return controlNetThresholdLow, controlNetThresholdHigh, controlNetTileSize, controlNetUpscaleMethod, controlNetTileSaveTiles

    def switchRenderEngine(
                self,
                type
        ):

        if type == "Diffusers":
            print("Switching to Diffusers render engine...")
            # TensorFlow Options
            legacyVersion = gr.Checkbox.update(visible = False)
            vPrediction = gr.Checkbox.update(visible = False)
            optimizerMethod = gr.Dropdown.update(visible = False)
            mixedPrecisionCheckbox = gr.Checkbox.update(visible = False)

            # Diffusers Options
            LoRAStrength = gr.Slider.update(visible = True)
            LoRAChoices = gr.CheckboxGroup.update(visible = True)
            tokenMergingStrength = gr.Slider.update(visible = True)
            CLIPSkip = gr.Slider.update(visible = True)
        elif type == "TensorFlow":
            print("Switching to TensorFlow Keras render engine...")
            # TensorFlow Options
            legacyVersion = gr.Checkbox.update(visible = True)
            vPrediction = gr.Checkbox.update(visible = True)
            optimizerMethod = gr.Dropdown.update(visible = True)
            mixedPrecisionCheckbox = gr.Checkbox.update(visible = True)

            # Diffusers Options
            LoRAStrength = gr.Slider.update(visible = False)
            LoRAChoices = gr.CheckboxGroup.update(visible = False)
            tokenMergingStrength = gr.Slider.update(visible = False)
            CLIPSkip = gr.Slider.update(visible = False)
        
        updatedWeights = self.dreamer.switchWeights(type)

        listOfModels = gr.Dropdown.update(choices = updatedWeights)
        
        return listOfModels, legacyVersion, vPrediction, LoRAStrength, LoRAChoices, optimizerMethod, mixedPrecisionCheckbox, tokenMergingStrength, CLIPSkip

    def switchResult(
            self,
            type
        ):

        if type == "Art":
            artResult = gr.Gallery.update(visible = True)
            videoResult = gr.Video.update(visible = False)
            return artResult, videoResult

        elif type == "Cinema":
            artResult = gr.Gallery.update(visible = False)
            videoResult = gr.Video.update(visible = True)
            return artResult, videoResult


class Render():
    def __init__(
            self,
            dreamer = None
    ):
        if dreamer == None:
            print("[red bold]FATAL ERROR[/red bold][yellow]\nNo dreamer class was given. Exiting Gradio GUI now[/yellow]")

        """
        Main Tools
        """

        # Prompts
        self.prompt = gr.Textbox(
            label = "Prompt - What should the AI create?"
        )

        self.negativePrompt = gr.Textbox(
            label = "Negative Prompt - What should the AI avoid when creating?"
        )

        # Creation Type

        self.creationType = gr.Radio(
            choices = ["Art", "Cinema"],
            value = dreamer.userSettings["creationType"],
            label = "Creation Type:"
        )

        # Start Button
        self.startButton = gr.Button("Start")

        self.startButton.style(
            full_width = True
        )

class ImageSettings():
    def __init__(
            self,
            dreamer = None,
            currentWeights = None,
            embeddingNames = None,
            LoRAs = None
    ):
        if dreamer == None:
            print("[red bold]FATAL ERROR[/red bold][yellow]\nNo dreamer class was given. Exiting Gradio GUI now[/yellow]")
        
        """
        Image Creation
        """

        """
        Settings
        """

        if dreamer.userSettings["defaultModel"] != "":
            defaultModelValue = currentWeights[currentWeights.index(self.dreamer.userSettings["defaultModel"])]
        else:
            defaultModelValue = currentWeights[0] if len(currentWeights) > 0 else None

        self.listOfModels = gr.Dropdown(
                    choices = currentWeights,
                    label = "Diffusion Model",
                    value = defaultModelValue
                )

        #TensorFlow
        self.legacyVersion = gr.Checkbox(
            label = "Use Legacy Stable Diffusion (1.4/1.5)",
            value = userSettingsBool(dreamer.userSettings["legacyVersion"]),
            visible = False
        )

        #TensorFlow
        self.vPrediction = gr.Checkbox(
            label = "Use SD 2.x-V",
            value = False,
            interactive = True,
            visible = False
        )

        # Height
        self.height = gr.Slider(
            minimum = 128,
            maximum = 1152,
            value = 512,
            step = 128,
            label = "Height"
        )

        # Width
        self.width = gr.Slider(
            minimum = 128,
            maximum = 1152,
            value = 512,
            step = 128,
            label = "Width"
        )

        # Batch Size

        self.batchSizeSelect = gr.Slider(
            minimum = 1,
            maximum = int(dreamer.userSettings["batchMax"]),
            value = int(dreamer.userSettings["defaultBatchSize"]),
            step = 1,
            label = "Batch Size"
        )

        # Steps
        self.steps = gr.Slider(
            minimum = 2,
            maximum = int(dreamer.userSettings["stepsMax"]),
            value = int(dreamer.userSettings["stepsMax"]) / 4,
            step = 1,
            label = "Sample Steps"
        )

        # Scale
        self.scale = gr.Slider(
            minimum = 2,
            maximum = int(dreamer.userSettings["scaleMax"]),
            value = 7.5,
            step = 0.1,
            label = "Guidance Scale"
        )

        # Seed
        self.seed = gr.Number(
            value = random.randint(0, 2 ** 31),
            label = "Seed",

        )

        sampleChoices = ["Basic", "DDIM", "DDPM", "DPM Solver", "DPM++ 2M","DPM++ 2M SDE","DPM++ 2S a Karras","DPM++ SDE","DPM2 a Karras","DPM2 Karras"
,"DPM2","DPM fast","Euler", "Euler A", "LMS", "LMS Karras", "PNDM"]

        self.sampleMethod = gr.Dropdown(
            choices = sampleChoices,
            label = "Sample Method",
            value = sampleChoices[0]
        )

        """
        Input Image
        """

        self.inputImage = gr.Image(
            label = "Input Image",
            type = "numpy"
        )

        # Input Image Strength
        self.inputImageStrength = gr.Slider(
            minimum = 0,
            maximum = 1,
            value = 0.5,
            step = 0.01,
            label = "0 = Don't change the image, 1 = ignore image entirely"
        )

        """
        Text Embeddings
        """

        self.useEmbeddings = gr.Checkbox(
            label = "Use Text Embeddings",
            value = False
        )

        self.embeddingChoices = gr.CheckboxGroup(
            choices = embeddingNames,
            label = "Select embeddings to include in model:"
        )

        """
        LoRA
        * Diffusers only
        """
        self.LoRAStrengthSlider = gr.Slider(
            label = "Strength",
            value = 0.5,
            minimum = 0,
            maximum = 1,
            step = 0.1
        )

        self.LoRAChoices = gr.CheckboxGroup(
            choices = LoRAs,
            label = "Select which LORA's to use"
        )

class ControlNet():
    def __init__(
            self,
            dreamer,
            controlNetWeights
    ):
        self.useControlNet = gr.Checkbox(
            label = "Use ControlNet",
            value = False
        )

        self.controlNetCache = gr.Checkbox(
            label = "Create cache",
            value = False
        )

        self.controlNetChoices = gr.Dropdown(
            choices = controlNetWeights,
            label = "ControlNet Model",
            value = controlNetWeights[0] if len(controlNetWeights) > 0 else None
        )

        self.controlNetStrengthSlider = gr.Slider(
            minimum = 0,
            maximum = 2,
            value = 1,
            step = 0.1,
            label = "Strength"
        )

        self.controlNetInputImage = gr.Image(
            label = "Input Image"
        )

        self.controlNetProcessedImage = gr.Image(
            label = "Processed Image"
        ).style(height = 896)

        self.guessMode = gr.Checkbox(
            label = "Guess Mode",
            value = False
        )

        processingOptions = ["None", "Canny", "HED", "Tile"]

        self.controlNetProcessingOptions = gr.Dropdown(
            choices = processingOptions,
            label = "Processing Option",
            value = processingOptions[0]
        )

        self.controlNetProcessButton = gr.Button("Preview Image Pre-Process")

        ## Canny Options

        self.controlNetThresholdLow = gr.Slider(
            minimum = 1,
            maximum = 255,
            value = 100,
            step = 1,
            label = "Low Threshold",
            visible = False
        )

        self.controlNetThresholdHigh = gr.Slider(
            minimum = 1,
            maximum = 255,
            value = 200,
            step = 1,
            label = "High Threshold",
            visible = False
        )

        ## Tile Options

        scaleOptions = [2, 4, 8]

        self.controlNetTileSize = gr.Dropdown(
            choices = scaleOptions,
            value = "4",
            label = "Scale",
            visible = False
        )

        upscalerOptions = ["BICUBIC", "ESRGAN-TensorFlow"]

        self.controlNetUpscaleMethod = gr.Dropdown(
            choices = upscalerOptions,
            value = "BICUBIC",
            label = "Upscale Method",
            visible = False
        )

        self.controlNetTileSaveTiles = gr.Checkbox(
            label = "Save Individual Tiles?",
            value = False
        )

class AdvancedSettings():
    def __init__(
            self,
            dreamer = None,
            VAEWeights = [],
            deviceChoices = []
    ):
        """
        Advanced Settings
        """

        renderFrameworkChoices = ["Diffusers", "TensorFlow"]

        self.renderFrameworkGradio = gr.Dropdown(
            choices = renderFrameworkChoices,
            label = "Render Engine",
            value = renderFrameworkChoices[0]
        )

        self.listOfVAEModels = gr.Dropdown(
                    choices = VAEWeights,
                    label = "VAE Options",
                    value = VAEWeights[0]
            )

        self.listOfDevices = self.createDeviceComponent(deviceChoices)

        optimizerChoices = ["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "RMSprop", "SGD"]

        #TensorFlow
        self.optimizerMethod = gr.Dropdown(
            choices = optimizerChoices,
            label = "Optimizer",
            value = optimizerChoices[5],
            visible = False
        )

        # Save user settings for prompt

        self.saveSettings = gr.Checkbox(
            label = "Save settings used for prompt creation?",
            value = userSettingsBool(dreamer.userSettings["saveSettings"])
        )

        # Mixed precision
        #TensorFlow
        self.mixedPrecisionCheckbox = gr.Checkbox(
            label = "Used mixed precision? (FP16)",
            value = userSettingsBool(dreamer.userSettings["mixedPrecision"]),
            visible = False
        )

        self.tokenMergingStrength = gr.Slider(
            label = "Token Margeing Strength %",
            value = 50,
            minimum = 0,
            maximum = 100,
            step = 1
        )

        self.CLIPSkip = gr.Slider(
            label = "CLIP Skip",
            value = 0,
            minimum = 0,
            maximum = 11,
            step = 1
        )
    
    def createDeviceComponent(
            self,
            devices
        ):
        deviceNames = []
        for device in devices:
            deviceNames.append(device['name'])
        
        if len(devices) > 1:
            active = True
        else:
            active = False

        radioComponent = gr.Radio(
            choices = deviceNames,
            value = deviceNames[0],
            label = "Render Device",
            interactive = active
        )

        return radioComponent

class Import():
    def __init__(
            self,
            dreamer
    ):
        """
        Import
        """

        self.importPromptLocation = gr.File(
            label = "Import Prior Prompt and settings for prompt",
            type = "file"
        )

        self.importPromptButton = gr.Button("Import prompt")

class Animation():
    def __init__(self):
        """
        Animation & Video
        """
        # Project Name
        self.projectName = gr.Textbox(
            value = "cinemaProject",
            label = "Name of the video - No spaces"
        )

        # FPS
        # Animated
        self.animatedFPS = gr.Dropdown(
            choices = [1,2,4,12,24,30,48,60],
            value = 12,
            label = "Animated Frames Per Second - 12 is standard animation"
        )
        # Final video
        self.videoFPS = gr.Dropdown(
            choices = [24,30,60],
            value = 24,
            label = "Video Frames Per Second - 24 is standard cinema"
        )

        # Total frames
        self.totalFrames = gr.Number(
            value = 48,
            label = "Total Frames",
        )

        # Starting frame

        self.startingFrame = gr.Number(
            value = 0,
            label = "Starting Frame Number"
        )

        # Seed behavior
        self.seedBehavior = gr.Dropdown(
            choices = ["Positive Iteration", "Negative Iteration", "Random Iteration", "Static Iteration"],
            value = "Positive Iteration",
            label = "Seed Behavior - How the seed changes from frame to frame"
        )

        # Save video
        self.saveVideo = gr.Checkbox(
            label = "Save result as a video?",
            value = True
        )

        self.reuseInputForVideo = gr.Checkbox(
            label = "Reuse initial input image?",
            value = False
        )

        self.reuseControlNetForVideo = gr.Checkbox(
            label = "Reuse initial ControlNet input image?",
            value = False
        )

        # Image Movement

        # X Translation
        self.xTranslate = gr.Slider(
            minimum = 0,
            maximum = 1000,
            value = 0,
            step = 1,
            label = "X Translation"
        )

        # Y Translation
        self.yTranslate = gr.Slider(
            minimum = 0,
            maximum = 1000,
            value = 0,
            step = 1,
            label = "Y Translation"
        )

        # Z Translation
        self.zTranslate = gr.Slider(
            minimum = 0,
            maximum = 1000,
            value = 200,
            step = 1,
            label = "Z Translation"
        )

        # X Rotation
        self.xRotation = gr.Slider(
            minimum = 0,
            maximum = 360,
            value = 90,
            step = 1,
            label = "X Rotation"
        )

        # Y Rotation
        self.yRotation = gr.Slider(
            minimum = 0,
            maximum = 360,
            value = 90,
            step = 1,
            label = "Y Rotation"
        )

        # Z Rotation
        self.zRotation = gr.Slider(
            minimum = 0,
            maximum = 360,
            value = 90,
            step = 1,
            label = "Z Rotation"
        )

        # Focal Length
        self.focalLength = gr.Slider(
            minimum = 0,
            maximum = 1000,
            value = 200,
            step = 10,
            label = "Focal Length"
        )

class DepthMapping():
    def __init__(self):
        self.inputImage = gr.Image(
            label = "Input Image",
            type = "filepath"
        )

        self.outputImage = gr.Image(
            label = "Output Image",
        )

        self.processImageButton = gr.Button("Process")

class ModelConversion():
    def __init__(self):
        self.saveModelName = gr.Textbox(
            label = "Model Name",
            value = "model"
        )

        self.saveModelButton = gr.Button("Save model")

        modelTypes = ["TensorFlow", "Diffusers", "Safetensors"]

        self.saveModelType = gr.Dropdown(
            choices = modelTypes,
            label = "Model Type",
            value = modelTypes[1]
        )

        self.pruneModelButton = gr.Button("Optimize Model")

class PromptEngineering():
    def __init__(
            self,
            embeddingNames = [],
            starterPrompt = [],
    ):
        # Create slots of prompt lists

        self.starterPrompts = self.createPromptComponents(starterPrompt)

        self.addPrompt = gr.Button("Add to prompt")

        # List of embeddings that were loaded into the program

        if len(embeddingNames) > 0:
            self.listOfEmbeddings = gr.Dropdown(
                        choices = embeddingNames,
                        label = "Text Embeddings",
                        value = embeddingNames[0]
                    )
        else:
            self.listOfEmbeddings = gr.Dropdown(
                        choices = None,
                        label = "Text Embeddings",
                        value = None
                    )
    
    def createPromptComponents(
            self,
            variable
        ):
        totalComponents = []
        for key in variable:
            component = gr.Dropdown(
                choices = variable[key],
                label = str(key),
                value = None
            )

            totalComponents.append(component)

        return totalComponents

class VideoTools():
    def __init__(
            self,
            dreamer = None
    ):
        # Video Tools
        self.convertToVideoButton = gr.Button("Convert to video")

        # input frames
        self.framesFolder = gr.Textbox(
            label = "Frames folder path",
            value = ""
        )

        # creations location
        self.creationsFolder = gr.Textbox(
            label = "Save Location",
            value = dreamer.userSettings["creationLocation"]
        )

        # video name
        self.videoFileName = gr.Textbox(
            label = "Video Name",
            value = "cinema"
        )

class PyTorchModelAnalysis():
    def __init__(
            self
    ):
        self.checkModelButton = gr.Button("Check Model")

        self.analyzeModelWeightsButton = gr.Button("Analyze Model Weights")

        analyzeThisModelChoices = ["Entire Model", "VAE", "Text Embeddings", "ControlNet"]

        self.analyzeThisModel = gr.Dropdown(
            choices = analyzeThisModelChoices,
            value = analyzeThisModelChoices[0],
            label = "What kind of model to analyze?"
        )

"""
Functions
"""

def userSettingsBool(
        setting
        ):
        if setting == "True":
            return True
        else:
            return False

def createLayout(
        gradioGUI = None
):
    """
    Main Layout
        Designed with Gradio's block system
    """

    with gr.Blocks(
        title = "MetalDiffusion"
    ) as demo:
        #Title
        gr.Markdown(
            "<center><span style = 'font-size: 32px'>MetalDiffusion</span><br><span style = 'font-size: 16px'>Stable Diffusion for MacOS<br>Intel Mac</span></center>"
        )

        with gr.Row():
            with gr.Column(scale = 3):
                with gr.Tab("Result"):
                            
                    gradioGUI.result.render()

                    gradioGUI.resultVideo.render()
                
                with gr.Tab("Preview"):
                    with gr.Tab("ControlNet"):
                        gradioGUI.controlNet.controlNetProcessedImage.render()
                    with gr.Tab("Video"):
                        gradioGUI.previewResult.render()

            with gr.Column(scale = 1):
                with gr.Tab("Render"):
                    # Start Button
                    gradioGUI.render.startButton.render()

                    # Creation type
                    gradioGUI.render.creationType.render()

                    # Prompts
                    gradioGUI.render.prompt.render()
                    gradioGUI.render.negativePrompt.render()
                
                with gr.Tab("Image Settings"):
                    #### Basic Settings
                    gr.Markdown("<center>Model Options</center>")

                    # Model Selection
                    gradioGUI.imageSettings.listOfModels.render()

                    # Legacy vs Contemporary Edition
                    # TensorFlow Only
                    gradioGUI.imageSettings.legacyVersion.render()
                    gradioGUI.imageSettings.vPrediction.render()

                    gr.Markdown("<center>Final Image Dimensions</center>")
                    # Width
                    gradioGUI.imageSettings.width.render()

                    # Height
                    gradioGUI.imageSettings.height.render()

                    gr.Markdown("<center>Generation Settings</center>")
                    # Batch Size
                    gradioGUI.imageSettings.batchSizeSelect.render()

                    ## Elementary Settings

                    # Steps
                    gradioGUI.imageSettings.steps.render()

                    # Scale
                    gradioGUI.imageSettings.scale.render()

                    with gr.Row():
                        # Seed
                        gradioGUI.imageSettings.seed.render()

                        newSeed = gr.Button("New Seed")
                    
                    # Sampler Method
                    gradioGUI.imageSettings.sampleMethod.render()
                
                with gr.Tab("Input Image"):
                    ## Input Image
                    gr.Markdown("<center><b><u>Input Image</b></u></center>Feed a starting image into the AI to give it inspiration")

                    gradioGUI.imageSettings.inputImage.render()

                    # Input Image Strength

                    gr.Markdown("Strength")

                    gradioGUI.imageSettings.inputImageStrength.render()
                
                with gr.Tab("Text Embeddings"):
                    # Text Embeddings
                    # useEmbeddings.render()

                    gradioGUI.imageSettings.embeddingChoices.render()
                
                with gr.Tab("LoRAs"):
                    # LoRAs
                    # Diffusers Only
                    gradioGUI.imageSettings.LoRAStrengthSlider.render()

                    gradioGUI.imageSettings.LoRAChoices.render()
                
                with gr.Tab("ControlNet"):
                    # Use ControlNet and Model Choice
                    with gr.Row():
                        with gr.Column():
                            gradioGUI.controlNet.useControlNet.render()
                            gradioGUI.controlNet.controlNetCache.render()
                            gradioGUI.controlNet.controlNetStrengthSlider.render()

                        with gr.Column():
                            gradioGUI.controlNet.controlNetChoices.render()
                            gradioGUI.controlNet.guessMode.render()

                    # ControlNet Input Image and Preview
                    gr.Markdown("<center>Input Image</center>")
                    with gr.Row():

                        gradioGUI.controlNet.controlNetInputImage.render()
                    
                    gr.Markdown("<center>Pre-Processing</center>")
                    # ControlNet Process Option and Preview Button
                    with gr.Row():
                        gradioGUI.controlNet.controlNetProcessingOptions.render()
                        gradioGUI.controlNet.controlNetProcessButton.render()
                    
                    with gr.Row():
                        # Canny Edge Options
                        gradioGUI.controlNet.controlNetThresholdLow.render()
                        gradioGUI.controlNet.controlNetThresholdHigh.render()

                    with gr.Row():
                        # Tile Options
                        with gr.Row():
                            gradioGUI.controlNet.controlNetTileSize.render()
                            gradioGUI.controlNet.controlNetUpscaleMethod.render()
                        with gr.Row():
                            gradioGUI.controlNet.controlNetTileSaveTiles.render()

                
                with gr.Tab("Animation"):

                    with gr.Tab("Settings"):

                        with gr.Row():

                            gradioGUI.animation.projectName.render()

                            gradioGUI.animation.saveVideo.render()

                        with gr.Row():
                            
                            gradioGUI.animation.animatedFPS.render()
                            
                            gradioGUI.animation.videoFPS.render()

                        with gr.Row():
                            gradioGUI.animation.totalFrames.render()

                            gradioGUI.animation.startingFrame.render()

                        with gr.Row():

                            gradioGUI.animation.seedBehavior.render()

                            with gr.Column():

                                gradioGUI.animation.reuseInputForVideo.render()

                                gradioGUI.animation.reuseControlNetForVideo.render()
                    
                    with gr.Tab("Camera Movement"):

                        gradioGUI.animation.xTranslate.render()

                        gradioGUI.animation.yTranslate.render()

                        gradioGUI.animation.zTranslate.render()

                        gradioGUI.animation.xRotation.render()

                        gradioGUI.animation.yRotation.render()

                        gradioGUI.animation.zRotation.render()

                        gradioGUI.animation.focalLength.render()
                
                with gr.Tab("Import"):

                    with gr.Tab("Creation"):
                        ## Import prior prompt and settings
                        gr.Markdown("<center><b><u>Import Creation</b></u></center>Import prior prompt and generator settings")
                        
                        gradioGUI.Import.importPromptLocation.render()
                        
                        gradioGUI.Import.importPromptButton.render()

                with gr.Tab("Tools"):
                    with gr.Tab("Depth Mapping"):
                        gr.Markdown("<center><b>Create a depth map from an image</b></center>")

                        with gr.Row():

                            gradioGUI.depthMapping.inputImage.render()

                            gradioGUI.depthMapping.outputImage.render()

                        gradioGUI.depthMapping.processImageButton.render()

                    with gr.Tab("Prompt Generator"):
                        gr.Markdown("Tools to generate useful prompts")

                        gradioGUI.promptEngineering.listOfEmbeddings.render()
                        
                        # Starter Prompts
                        for item in gradioGUI.promptEngineering.starterPrompts:
                            item.render()

                        gradioGUI.promptEngineering.addPrompt.render()
                    with gr.Tab("Model Conversion"):

                        gr.Markdown("<center><b>Save Current model as either <bold>Diffusers</bold> or <bold>Keras</bold> '.h5' weights</b><br>Useful for converting PyTorch/Safetensors to Diffusers/Keras foramt.</center>")

                        with gr.Row():

                            with gr.Column():

                                gradioGUI.modelConversion.saveModelName.render()

                            with gr.Column():
                                gradioGUI.modelConversion.saveModelType.render()
                                gradioGUI.modelConversion.saveModelButton.render()
                        
                        #gradioGUI.modelConversion.pruneModelButton.render()
                    
                    with gr.Tab("Video Tools"):
                        gr.Markdown("<center>Image sequence to video</center>")
                        with gr.Row():

                            with gr.Column():

                                gradioGUI.videoTools.framesFolder.render()

                                gradioGUI.videoTools.creationsFolder.render()
                            
                            with gr.Column():

                                gradioGUI.videoTools.videoFileName.render()

                                gradioGUI.videoTools.convertToVideoButton.render()
                    
                    with gr.Tab("PyTorch Model Analysis"):
                        gr.Markdown("<center>What makes a pytroch model tick?</center>")

                        gradioGUI.modelAnalysis.checkModelButton.render()

                        gradioGUI.modelAnalysis.analyzeModelWeightsButton.render()

                        gradioGUI.modelAnalysis.analyzeThisModel.render()

                with gr.Tab("Advanced Settings"):

                    with gr.Row():
                        gradioGUI.advancedSettings.renderFrameworkGradio.render()

                    with gr.Row():

                        # VAE Selection
                        gradioGUI.advancedSettings.listOfVAEModels.render()

                        # Optimizer
                        # TensorFlow Only
                        gradioGUI.advancedSettings.optimizerMethod.render()

                    with gr.Row():

                        # Save settings used for creation?
                        gradioGUI.advancedSettings.saveSettings.render()

                        # Mixed Precision
                        # TensorFlow Only
                        gradioGUI.advancedSettings.mixedPrecisionCheckbox.render()
                    
                    with gr.Row():

                        # Token Merging
                        # Diffusers Only
                        gradioGUI.advancedSettings.tokenMergingStrength.render()
                    
                    with gr.Row():

                        # CLIP Skip
                        # Diffusers Only
                        gradioGUI.advancedSettings.CLIPSkip.render()

                    with gr.Row():
                        # Device Selection
                        gradioGUI.advancedSettings.listOfDevices.render()

                with gr.Tab("About"):
                    gr.Markdown("<center><b>MetalDiffusion</b><br>Created by AJ Young<br>Designed for Intel Mac's<br>Special Thank you to Divum Gupta<br><br>Compel Syntax: <a href = 'https://github.com/damian0815/compel/blob/main/doc/syntax.md'>Syntax link</a>")

        with gr.Row():

            gradioGUI.endProgramButton.render()

        ## Event actions

        ## Groups

        controlNetPreProcessingInputs = [gradioGUI.controlNet.controlNetInputImage, gradioGUI.controlNet.controlNetProcessingOptions, gradioGUI.controlNet.controlNetThresholdLow, gradioGUI.controlNet.controlNetThresholdHigh]

        imageTransformOptions = [gradioGUI.imageSettings.inputImage, gradioGUI.animation.xTranslate, gradioGUI.animation.yTranslate, gradioGUI.animation.zTranslate, gradioGUI.animation.xRotation, gradioGUI.animation.yRotation, gradioGUI.animation.zRotation, gradioGUI.animation.focalLength]

        """
        Top Row
        """

        # When start button is pressed
        gradioGUI.render.startButton.click(
            fn = gradioGUI.dreamer.create,
            inputs = [
                gradioGUI.render.creationType,
                gradioGUI.render.prompt,
                gradioGUI.render.negativePrompt,
                gradioGUI.imageSettings.width,
                gradioGUI.imageSettings.height,
                gradioGUI.imageSettings.scale,
                gradioGUI.imageSettings.steps,
                gradioGUI.imageSettings.seed,
                gradioGUI.imageSettings.inputImage,
                gradioGUI.imageSettings.inputImageStrength,
                gradioGUI.imageSettings.listOfModels,
                gradioGUI.imageSettings.batchSizeSelect,
                gradioGUI.advancedSettings.saveSettings,
                gradioGUI.animation.projectName,
                gradioGUI.animation.animatedFPS,
                gradioGUI.animation.videoFPS,
                gradioGUI.animation.totalFrames,
                gradioGUI.animation.seedBehavior,
                gradioGUI.animation.saveVideo,
                gradioGUI.animation.xTranslate, gradioGUI.animation.yTranslate, gradioGUI.animation.zTranslate,
                gradioGUI.animation.xRotation, gradioGUI.animation.yRotation, gradioGUI.animation.zRotation,
                gradioGUI.animation.focalLength,
                gradioGUI.animation.startingFrame,
                gradioGUI.imageSettings.legacyVersion,
                gradioGUI.advancedSettings.listOfVAEModels,
                gradioGUI.imageSettings.embeddingChoices,
                gradioGUI.advancedSettings.mixedPrecisionCheckbox,
                gradioGUI.imageSettings.sampleMethod,
                gradioGUI.advancedSettings.optimizerMethod,
                gradioGUI.advancedSettings.listOfDevices,
                gradioGUI.controlNet.useControlNet,
                gradioGUI.controlNet.controlNetChoices,
                gradioGUI.controlNet.controlNetProcessingOptions,
                gradioGUI.controlNet.controlNetInputImage,
                gradioGUI.controlNet.guessMode,
                gradioGUI.controlNet.controlNetStrengthSlider,
                gradioGUI.controlNet.controlNetCache,
                gradioGUI.controlNet.controlNetThresholdLow,
                gradioGUI.controlNet.controlNetThresholdHigh,
                gradioGUI.controlNet.controlNetTileSize,
                gradioGUI.controlNet.controlNetUpscaleMethod,
                gradioGUI.controlNet.controlNetTileSaveTiles,
                gradioGUI.imageSettings.vPrediction,
                gradioGUI.animation.reuseInputForVideo,
                gradioGUI.animation.reuseControlNetForVideo,
                gradioGUI.advancedSettings.renderFrameworkGradio,
                gradioGUI.imageSettings.LoRAChoices,
                gradioGUI.imageSettings.LoRAStrengthSlider,
                gradioGUI.advancedSettings.tokenMergingStrength,
                gradioGUI.advancedSettings.CLIPSkip
            ],
            outputs = [gradioGUI.result, gradioGUI.resultVideo]
        )

        # When creation type is selected
        gradioGUI.render.creationType.change(
            fn = gradioGUI.switchResult,
            inputs = gradioGUI.render.creationType,
            outputs = [gradioGUI.result, gradioGUI.resultVideo]
        )

        """
        Image Creation Settings
        """
        
        # When new seed is pressed
        newSeed.click(
            fn = gradioGUI.randomSeed,
            inputs = None,
            outputs = gradioGUI.imageSettings.seed,
        )

        """
        Advanced Settings
        """

        gradioGUI.advancedSettings.renderFrameworkGradio.change(
            fn = gradioGUI.switchRenderEngine,
            inputs = gradioGUI.advancedSettings.renderFrameworkGradio,
            outputs = [
                gradioGUI.imageSettings.listOfModels,
                gradioGUI.imageSettings.legacyVersion,
                gradioGUI.imageSettings.vPrediction,
                gradioGUI.imageSettings.LoRAStrengthSlider,
                gradioGUI.imageSettings.LoRAChoices,
                gradioGUI.advancedSettings.optimizerMethod,
                gradioGUI.advancedSettings.mixedPrecisionCheckbox,
                gradioGUI.advancedSettings.tokenMergingStrength,
                gradioGUI.advancedSettings.CLIPSkip
            ]
        )

        """
        ControlNet
        """

        gradioGUI.controlNet.controlNetProcessButton.click(
            fn = controlNetUtilities.previewProcessControlNetImage,
            inputs = controlNetPreProcessingInputs,
            outputs = gradioGUI.controlNet.controlNetProcessedImage
        )
        
        gradioGUI.controlNet.controlNetProcessingOptions.change(
            fn = gradioGUI.switchControlNetOptions,
            inputs = gradioGUI.controlNet.controlNetProcessingOptions,
            outputs = [
                gradioGUI.controlNet.controlNetThresholdLow,
                gradioGUI.controlNet.controlNetThresholdHigh,
                gradioGUI.controlNet.controlNetTileSize,
                gradioGUI.controlNet.controlNetUpscaleMethod,
                gradioGUI.controlNet.controlNetTileSaveTiles
            ]
        )

        """
        Video
        """

        for imageTransform in imageTransformOptions:
            imageTransform.change(
                fn = gradioGUI.previewImageTransform,
                inputs = imageTransformOptions,
                outputs = gradioGUI.previewResult
            )

        """
        Import
        """

        # When import button is pressed
        gradioGUI.Import.importPromptButton.click(
            fn = readWriteFile.importCreationSettings,
            inputs = gradioGUI.Import.importPromptLocation,
            outputs = [
                gradioGUI.render.prompt,
                gradioGUI.render.negativePrompt,
                gradioGUI.imageSettings.width,
                gradioGUI.imageSettings.height,
                gradioGUI.imageSettings.scale,
                gradioGUI.imageSettings.steps,
                gradioGUI.imageSettings.seed,
                gradioGUI.imageSettings.listOfModels,
                gradioGUI.imageSettings.batchSizeSelect,
                gradioGUI.imageSettings.inputImageStrength,
                gradioGUI.animation.animatedFPS,
                gradioGUI.animation.videoFPS,
                gradioGUI.animation.totalFrames,
                gradioGUI.animation.seedBehavior,
                gradioGUI.animation.yRotation,
                gradioGUI.animation.focalLength,
                gradioGUI.animation.xTranslate,
                gradioGUI.animation.yTranslate,
                gradioGUI.controlNet.controlNetChoices,
                gradioGUI.controlNet.controlNetStrengthSlider,
                gradioGUI.imageSettings.sampleMethod
            ]
        )

        """
        Tools
        """

        """
        Tools > Depth Mapping
        """

        gradioGUI.depthMapping.processImageButton.click(
            fn = generateDepthMap,
            inputs = [gradioGUI.depthMapping.inputImage],
            outputs = gradioGUI.depthMapping.outputImage
        )

        """
        Tools > Prompt Generator
        """

        # When add prompt is pressed
        gradioGUI.promptEngineering.addPrompt.click(
            fn = gradioGUI.addToPrompt,
            inputs = [gradioGUI.render.prompt, gradioGUI.promptEngineering.listOfEmbeddings, gradioGUI.promptEngineering.starterPrompts[0], gradioGUI.promptEngineering.starterPrompts[1], gradioGUI.promptEngineering.starterPrompts[2], gradioGUI.promptEngineering.starterPrompts[3], gradioGUI.promptEngineering.starterPrompts[4]],
            outputs = [gradioGUI.render.prompt, gradioGUI.promptEngineering.starterPrompts[0], gradioGUI.promptEngineering.starterPrompts[1], gradioGUI.promptEngineering.starterPrompts[2], gradioGUI.promptEngineering.starterPrompts[3], gradioGUI.promptEngineering.starterPrompts[4]]
        )

        # Save Model

        gradioGUI.modelConversion.saveModelButton.click(
            fn = gradioGUI.saveModel,
            inputs = [
                gradioGUI.modelConversion.saveModelName,
                gradioGUI.imageSettings.listOfModels,
                gradioGUI.imageSettings.legacyVersion,
                gradioGUI.modelConversion.saveModelType
            ],
            outputs = None
        )

        # Check Model

        gradioGUI.modelAnalysis.checkModelButton.click(
            fn = modelWrangler.checkModel,
            inputs = [gradioGUI.imageSettings.listOfModels, gradioGUI.imageSettings.legacyVersion],
            outputs = None
        )

        gradioGUI.videoTools.convertToVideoButton.click(
            fn = gradioGUI.dreamer.deliverCinema,
            inputs = [gradioGUI.videoTools.framesFolder, gradioGUI.videoTools.creationsFolder, gradioGUI.videoTools.videoFileName],
            outputs = gradioGUI.resultVideo
        )

        gradioGUI.modelAnalysis.analyzeModelWeightsButton.click(
            fn = modelWrangler.analyzeModelWeights,
            inputs = [
                gradioGUI.imageSettings.listOfModels,
                gradioGUI.advancedSettings.listOfVAEModels,
                gradioGUI.promptEngineering.listOfEmbeddings,
                gradioGUI.modelAnalysis.analyzeThisModel,
            ],
            outputs = None
        )

        """
        Bottom Row
        """

        gradioGUI.endProgramButton.click(
            fn = gradioGUI.endProgram,
            inputs = None,
            outputs = None
        )
    
    return demo

"""
This section is for running the script on it's own
"""

if __name__ == "__main__":
    print("Starting [bold]debugging[/bold] for gradioGUI.py")
    gradioGUIObject = gradioGUIHandler()