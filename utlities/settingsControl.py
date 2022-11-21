import os, fnmatch, sys, configparser

### Global variables

### Functions

def loadSettings(fileLocation, returnType = 0):
    try:
        # Load config file
        print("Loading user preferences...")
        config = configparser.ConfigParser() # use configparser to read the config file
        config.read(fileLocation) # read the file

        # Create list for returning the values

        if returnType == 0:

            configSettings = []

            for section in config.sections():
                for option, value in config.items(section):
                    configSettings.append(value)
        elif returnType == 1:

            configSettings = {}

            for section in config.sections():
                currentList = []
                configSettings[section] = []
                for option, value in config.items(section):
                    currentList.append(value)
                    configSettings.update( {section : currentList} )
        
        print("...preferences loaded!")
        return configSettings

    except Exception as e:
        print(e)
        print("File does not exist!")
        return False

def createUserPreferences(fileLocation, settings):
    configFile = configparser.ConfigParser()

    configFile["Settings"] = {
        "stepsMax": settings[0],
        "scaleMax": settings[1],
        "batchMax": settings[2],
        "defaultBatchSize" : settings[3],
        "modelsLocation" : settings[4],
        "defaultModel" : settings[5],
        "maxMemory" : settings[6],
        "creationLocation" : settings[7]
    }

    with open(fileLocation, 'w') as conf:
        configFile.write(conf)