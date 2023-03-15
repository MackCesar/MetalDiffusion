import os, fnmatch, sys, configparser

### Global variables

### Functions

def loadSettings(fileLocation, returnType = 0):
    try:
        # Load config file
        config = configparser.ConfigParser() # use configparser to read the config file
        config.read(fileLocation) # read the file

        # Create list for returning the values

        if returnType == 0: # Load user settings
            print("Loading user preferences...")

            configSettings = config["Settings"]

        elif returnType == 1: # Load Prompts from Prompt Generator file
            print("Loading prompts...")

            configSettings = {}

            for section in config.sections():
                currentList = []
                configSettings[section] = []
                for option, value in config.items(section):
                    currentList.append(value)
                    configSettings.update( {section : currentList} )
        
        print("...loaded!")
        return configSettings

    except Exception as e:
        print(e)
        print("File does not exist!")
        return False

def createUserPreferences(fileLocation):
    print("No user preferences found.\nCreating new preferences file.")
    configFile = configparser.ConfigParser()

    # Factory Settings
    configFile["Settings"] = {
        "stepsMax": 100,
        "scaleMax": 20,
        "batchMax": 4,
        "defaultBatchSize" : 1,
        "modelsLocation" : "models/",
        "defaultModel" : "",
        "creationLocation" : "creations/",
        "creationType" : "Art",
        "legacyVersion" : "True",
        "saveSettings" : "True",
        "VAEModelsLocation" : "models/VAE/",
        "EmbeddingsLocation" : "models/embeddings/",
        "mixedPrecision" : "False"
    }

    with open(fileLocation, 'w') as conf:
        configFile.write(conf)
    
    print("Type of variable: ")
    print(type(configFile["Settings"]))

    return configFile["Settings"]