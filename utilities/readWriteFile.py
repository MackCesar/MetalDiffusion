### System Modules
import os
import sys
import random

def writeToFile(path, text):
    try:
        file = open(path, "w")

        for i, item in enumerate(text):
            text[i] = str(item) + "\n"

        file.writelines(text)
        file.close()

        return True
    except Exception as e:
        print(e)
        return False

def readFromFile(path):
    #Expecting path to be a file type object
    file = open(path.name, "r")

    data = file.read()

    list = data.split("\n")

    for item in range(len(list)):
        # [0]Prompt, [1]NegativePrompt, [13]SeedBehavior
        # [16]xTranslation, [17]yTranslation
        # [2]Width, [3]Height, [5]Steps, [8]Batch Size, [10]AnimatedFPS
        # [11]VideoFPS, [12]Total Frames
        if item is 2 or item is 3 or item is 5 or item is 8 or item is 10 or item is 11 or item is 12:
            list[item] = int(list[item])
        # [4]Scale, [9]Input Image Strength, [14]Angle, [15]Zoom
        if item is 4 or item is 9 or item is 14 or item is 15:
            list[item] = float(list[item])
        # [6]Seed
        if item is 6:
            list[item] = int(float(list[item]))

    file.close()

    return list