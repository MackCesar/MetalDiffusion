### System Modules
import os
import sys
import random

def writeToFile(path, text, dictionaryToFind = None):
    try:
        file = open(path, "w")

        if dictionaryToFind is not None and dictionaryToFind != "All":
            data = text
            text = ""
            i = 0
            for key in data[dictionaryToFind].items():
                i = i +1
                text = text + str(i) + " " + str(key[0]) + "\n" # Layer
                text = text + str(key[1]) + "\n" # Value
        elif dictionaryToFind is None:
            for i, item in enumerate(text):
                text[i] = str(item) + "\n"
        elif dictionaryToFind == "All":
            data = text
            text = ""
            for key, value in data.items():
                text = text + str(key) + "\n"
        elif dictionaryToFind == "embedding":
            data = text
            text = ""
            for key, value in data.items():
                text = text + str(value) + ":" + str(value) + "\n"
        else:
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

def readFromFile(path):
    #Expecting path to be a file type object
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
        if item is 2 or item is 3 or item is 5 or item is 8 or item is 10 or item is 11 or item is 12:
            if creationList[item] == "NODATA":
                creationList[item] = 1
            else:
                creationList[item] = int(creationList[item])
        # [4]Scale, [9]Input Image Strength, [14]Angle, [15]Zoom, [19]ControlNet Strength
        if item is 4 or item is 9 or item is 14 or item is 15 or item is 19:
            if creationList[item] == "NODATA":
                creationList[item] = 0.0
            elif creationList[item] == "slider":
                creationList[item] == "1"
            else:
                creationList[item] = float(creationList[item])
        # [6]Seed
        if item is 6:
            if creationList[item] == "NODATA":
                creationList[item] = 12345
            else:
                creationList[item] = int(float(creationList[item]))

    file.close()

    return creationList