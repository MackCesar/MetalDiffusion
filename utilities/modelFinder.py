### System modules
import os
import torch as torch
from .consoleUtilities import color
from . import readWriteFile as readWriteFile

# Find model/weights for program to use
def findModels(
        path,
        type,
        getHash = False
    ):
    if type == "":
        type = "Keras .h5"
    print("\nSearching for:", type)
    directory = os.listdir(path.rstrip()) # local variable for a list of the files in the directory
    currentList = [] # local variable list for found files with a default already added
    for file in directory:
        if type != "Keras .h5":
            # For all types EXCEPT .h5
            if file.endswith(type):
                # Prints only type of file present in the folder
                print(" ",file," found!")
                if getHash is True:
                    hash = " [" + modelHash(path + "/" + file) + "]"
                else:
                    hash = ""
                finalName = str(file) + hash
                # print(finalName)
                currentList.append(finalName)
        else:
            if "VAE" not in file and "embeddings" not in file and "controlnets" not in file and "." not in file:
                # Skip 'VAE', 'embeddings', and hidden '.' folders
                print(" ",file," found!")
                currentList.append(file)
    
    print("...finished!")
    return currentList # returns the final list of files

# Get the hash of a model
def modelHash(filename):
    try:
        with open(filename, "rb") as file:
            print("opening file\nimporting hashlib")
            import hashlib
            m = hashlib.sha256()

            #hashing file
            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

def analyzeModelWeights(model, VAE, textEmbeddings, whichModel):
    global userSettings

    if whichModel == "VAE":
        thePatient = VAE
        filePath = userSettings["VAEModelsLocation"]
        dictionaryToFind = "state_dict"
        fileType = ".ckpt"
    elif whichModel == "Text Embeddings":
        thePatient = textEmbeddings
        filePath = userSettings["EmbeddingsLocation"]
        dictionaryToFind = "embedding"
        if "pt" in thePatient:
            fileType = ".pt"
        else:
            fileType = ".bin"
    elif whichModel == "ControlNet":
        thePatient = model
        filePath = userSettings["modelsLocation"]
        dictionaryToFind = "All"
        fileType = ".pth"
    elif whichModel == "Entire Model":
        thePatient = model
        filePath = userSettings["modelsLocation"]
        dictionaryToFind = "state_dict"
        if ".ckpt" in thePatient:
            fileType = ".ckpt"
        else:
            print("\nUnable to analyze model.\n",thePatient,"was given, which is not a pytorch .ckpt file. Most likely a ControlNet model was given")
            return
    print("\nAnalyzing model weights for: ", thePatient)

    print("...analyzing...")

    pytorchWeights = torch.load(filePath + thePatient, map_location = "cpu")

    """for key, value in pytorchWeights.items():
            valueCheck = str(value)
            if "tensor" in valueCheck:
                print(str(key))
                for token, vector in value.items():
                    print(vector)
                    print(vector.detach().numpy())
                print(value.numpy())
                pytorchWeights.append(str(key))
                pytorchWeights.append(value.numpy())

    print(pytorchWeights)"""
    print("...done!")

    print("Saving analysis...")
    
    if readWriteFile.writeToFile(filePath + thePatient.replace(fileType,"-analysis.txt"), pytorchWeights, dictionaryToFind):
        print("...done!")
    
    del pytorchWeights

def checkModel(selectedModel, legacy):

    global dreamer

    dreamer.pytorchModel = selectedModel

    dreamer.legacy = legacy

    # Have we compiled any models already?
    if dreamer.generator is None:
        dreamer.compileDreams()
    
    # Set local variables
    model = dreamer.generator

    print("\nText Encoder Model Summary")
    model.text_encoder.summary()

    print("\nDiffusion Model Summary")
    model.diffusion_model.summary()
    try:
        model.diffusion_model.layers[3].summary()
    except Exception as e:
        print(e)

    print("\nDecoder Model Summary")
    model.decoder.summary()

    print("\nEncoder Model Summary")
    model.encoder.summary()

def saveModel(
    name = "model",
    type = ""
):
    global dreamer

    # Have we compiled any models already?
    if dreamer.generator is None:
        print("Compiling models")
        dreamer.compileDreams()
    
    # Set local variables
    model = dreamer.generator
    fileName = []

    for modelType in ["text_encoder", "diffusion_model", "decoder", "encoder"]:
        fileName.append(name + "_" + modelType + type)

    # Load/create folder to save frames in
    path = f"models/{name}"
    if not os.path.exists(path): #If it doesn't exist, create folder
        os.makedirs(path)
    
    # Save Text Encoder
    print("\nSaving model as:\n",fileName[0])
    model.text_encoder.save(path + fileName[0])
    print(color.GREEN,"Model saved!",color.END)

    # Save Diffusion Model
    print("\nSaving model as:\n",fileName[1])
    model.diffusion_model.save(path + fileName[1])
    print(color.GREEN,"Model saved!",color.END)

    # Save Decoder
    print("\nSaving model as:\n",fileName[2])
    model.decoder.save(path + fileName[2])
    print(color.GREEN,"Model saved!",color.END)

    # Save Encoder
    print("\nSaving model as:\n",fileName[3])
    model.encoder.save(path + fileName[3])
    print(color.GREEN,"Model saved!",color.END)