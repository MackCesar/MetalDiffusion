### System Modules
import os
import sys
import random

### Image Modules
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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

"""
ReadWriteFile

This general purpose module allows the user of MetalDiffusion to do some file operations.

Currently imports:
.txt files, .png files

Currently writes:
.txt files detailng the structure of model weights

"""

def importCreationSettings(filePath):
    if isinstance(filePath, str) is False:
        print("\nReceived Gradio.File object. Getting file path:")
        filePath = filePath.name
        print(filePath)
    if ".txt" in filePath:
        print("\nLoading creation settings from a text file!")
        settings = importFromTextFile(filePath)
    if ".png" in filePath:
        print("\nLoading creation settings from an image file!")
        settings = importFromPNGFile(filePath)
    
    return settings

def importFromPNGFile(filePath):
    # Create the empty settings variable
    creationList = []
    for _ in range(21):
        creationList.append(None)
    
    creationList[0] = "prompt"                  # Prompt
    creationList[1] = "negative prompt"         # Negative Prompt
    creationList[2] = int(512)                  # Width
    creationList[3] = int(512)                  # Height
    creationList[4] = float(7.5)                # CFG Scale
    creationList[5] = int(25)                   # Steps
    creationList[6] = int(1337)                 # Seed
    creationList[7] = None                      # Weights
    creationList[8] = int(1)                    # Batch Size
    creationList[9] = float(0.5)                # Input Image Strength
    creationList[10] = "12"                     # Animated FPS
    creationList[11] = "24"                     # Video FPS
    creationList[12] = "48"                     # Total Frames
    creationList[13] = "Positive Iteration"     # Video Seed Behavior
    creationList[14] = 90                       # Video Angle
    creationList[15] = 200                      # Video Zoom
    creationList[16] = "0"                      # Video X Translate
    creationList[17] = "0"                      # Video Y Translate
    creationList[18] = None                     # ControlNet Model
    creationList[19] = [1]                      # ControlNet Strength
    creationList[20] = "DDIM"                   # Sample Method
    
    # Load the file
    if isinstance(filePath, str):
        # String which is a path to the file
        file = Image.open(filePath)
    else:
        # Default is a file object from Gradio.File
        file = Image.open(filePath)
    
    # Check if the image has PNG format
    if file.format != "PNG":
        print("The file is not in PNG format.")
        return
    
    # Load the EXIF/Metadata
    file.load()

    # Get the PNG info dictionary
    pngInfo = file.info

    if pngInfo is None:
        print("No PNG info found in the image.")
        return
    
    # Parse and process the PNG info
    for key, value in pngInfo.items():
        print(f"{key}: {value}")
        if key == "prompt": creationList[0] = value
        if key == "negative prompt": creationList[1] = value
        if key == "seed": creationList[6] = int(float(value))
        if key == "CFG scale": creationList[4] = float(value)
        if key == "steps": creationList[5] = int(value)
        if key == "input image strength": creationList[9] = float(value)
        if key == "controlNet strength": creationList[19] = float(value)
        if key == "model": creationList[7] = value
        if key == "batch size": creationList[8] = int(value)
        if key == "sampler": creationList[20] = value
    
    # Add info about the image to the CreationList
    creationList[2] = file.width
    creationList[3] = file.height
    
    # Close the image
    file.close()

    return creationList

def importFromTextFile(path):
    if isinstance(path, str):
        # String which is a path to the file
        file = open(path, "r")
        data = file.read()
    else:
        # Default is a file object from Gradio.File
        file = open(path.name, "r")
        data = file.read()

    creationList = data.split("\n")

    if len(creationList) < 20:
        print("\nImporting data from an old structure, some data may be incorrect or missing")
        missingItems = 20 - len(creationList)
        for item in range(missingItems):
            creationList.append("NODATA")

    for item in range(len(creationList)):
        # [0]Prompt, [1]NegativePrompt, [13]SeedBehavior
        # [16]xTranslation, [17]yTranslation
        # [18]ControlNet Weights

        # Strings ^

        # [2]Width, [3]Height, [5]Steps, [8]Batch Size, [10]AnimatedFPS
        # [11]VideoFPS, [12]Total Frames
        if item == 2 or item == 3 or item == 5 or item == 8 or item == 10 or item == 11 or item == 12:
            if creationList[item] == "NODATA":
                creationList[item] = 1
            else:
                creationList[item] = int(creationList[item])
        # [4]Scale, [9]Input Image Strength, [14]Angle, [15]Zoom
        if item == 4 or item == 9 or item == 14 or item == 15:
            if creationList[item] == "NODATA":
                creationList[item] = 0.0
            elif creationList[item] == "slider":
                creationList[item] == "1"
            else:
                creationList[item] = float(creationList[item])
        # [6]Seed
        if item == 6:
            if creationList[item] == "NODATA":
                creationList[item] = 12345
            else:
                creationList[item] = int(float(creationList[item]))
        #[19]ControlNet Strength
        if item == 19:
            if creationList[item] == "NODATA":
                creationList[item] = 0.0
            else:
                creationList[item] = list(creationList[item])

    file.close()

    return creationList

def writeToFile(path, text, dictionaryToFind = None):
    try:
        file = open(path, "w")

        if dictionaryToFind is not None and dictionaryToFind != "All":
            # Specific Model Weight Analysis
            data = text
            text = ""
            i = 0
            for key in data[dictionaryToFind].items():
                i = i +1
                text = text + str(i) + " " + str(key[0]) + "\n" # Layer
                text = text + str(key[1]) + "\n" # Value
        elif dictionaryToFind is None:
            # Creation settings
            for i, item in enumerate(text):
                text[i] = str(item) + "\n"
        elif dictionaryToFind == "All":
            # Entire Model Weight Analysis
            data = text
            text = ""
            for key, value in data.items():
                text = text + str(key) + "\n"
        elif dictionaryToFind == "embedding":
            # Embedding Specific Weight Analysis
            data = text
            text = ""
            for key, value in data.items():
                text = text + str(value) + ":" + str(value) + "\n"
        else:
            # I don't remember what this was for :)
            data = text
            text = ""
            for key, value in data.items():
                text = text + str(key) + " " + str(value) + "\n"

        file.writelines(text)
        file.close()

        return True
    except Exception as e:
        print("Error in writing file!\n",e)
        return False

def writeToXMLFile(path, text):
    print("I'm writing to XML! IDK why!")

"""
This section is for running the code independently of the entire program
"""
if __name__ == "__main__":
    print("Welcome to readWriteFile!")

    print("Testing out file types")

    textFile = "creations/848710974.0.txt"
    imageFile = "creations/8487109741.png"

    importCreationSettings(textFile)

    importCreationSettings(imageFile)