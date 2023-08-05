"""
Model Wrangler
--------------

Howdy, partner! Are your models and weights roaming free in your pastures? Then let Model Wrangler take care of corralling them!

Currently supports:
    + Stable Diffusion Models and their variants:
        .safetensors
        Diffusers, expecting a folder
        .ckpt - pytorch
        TensorFlow old weights file, expecting a folder of .h5's
    + VAE models
        .ckpt - pytroch
        Diffusers, expecting a folder
    + Text Embeddings
        .pt
        .bin
    + ControLNet
        .safetensors
        .pth
        .ckpt
        Diffusers, expecting a folder

"""

### System modules
import os
import torch as torch
from .consoleUtilities import color
from . import readWriteFile as readWriteFile

### Console GUI
from rich import print, box
from rich.panel import Panel
from rich.text import Text

### Memory Management
import gc

def findAllWeights(
        userSettings = None
):
    if userSettings == None:
        print("[bold red]Fatal Error![/bold red]:\n[yellow]No userSettings variable given. Unable to search!")
        return None
    print(
        Panel(
            ":horse_racing: :horse_racing: :horse_racing: :horse_racing: Yeeeeeehawwww, let's round up 'em weights!:horse_racing: :horse_racing: :horse_racing: :horse_racing: :horse_racing:",
            title = ":cowboy_hat_face:Model Wrangler:cowboy_hat_face:",
            box = box.ASCII,
            style = "yellow"
            )
        )
    allWeights = {
        "diffusers": [],
        "tensorflow": [],
        "safetensors": [],
        "ckpt": [],
        "huggingFace": []
    }
    print("\n...loading [bold]Diffusers[/bold] compatible weights...")
    allWeights["diffusers"] = findModels(userSettings["modelsLocation"] + "/diffusers/", "")
    print("\n...loading [bold]TensorFlow[/bold] compatible weights...")
    allWeights["tensorflow"] = findModels(userSettings["modelsLocation"] + "/tensorflow/", "")
    print("\n...loading [bold]universal[/bold] weights...")
    allWeights["safetensors"] = findModels(userSettings["modelsLocation"] + "/safetensors/", ".safetensors")
    allWeights["ckpt"] = findModels(userSettings["modelsLocation"] + "/ckpt/", ".ckpt")
    allWeights["huggingFace"] = ["runwayml/stable-diffusion-v1-5"]

    print("\nSearching for [blue]VAE[/blue] models/weights...")
    VAEWeights = findModels(userSettings["VAEModelsLocation"], ".ckpt")
    VAEWeights.sort()
    VAEWeights.insert(0,"Original")

    print("\nSearching for [blue]text embeddings[/blue]...")
    embeddingWeights = findModels(userSettings["EmbeddingsLocation"], ".pt")
    embeddingWeights.extend(findModels(userSettings["EmbeddingsLocation"], ".bin"))
    embeddingWeights.sort()
    # Store names with <> around them for prompt generator
    embeddingNames = embeddingWeights.copy()
    for index, name in enumerate(embeddingNames):
        if "pt" in name:
            embeddingNames[index] = "<" + name.replace(".pt","") + ">"
            embeddingNames[index] = embeddingNames[index].lower()
        if "bin" in name:
            embeddingNames[index] = "<" + name.replace(".bin","") + ">"
            embeddingNames[index] = embeddingNames[index].lower()
    # Add the filepath as the first index to the embeddingWeights variable
    embeddingWeights.insert(0, userSettings["EmbeddingsLocation"])

    print("\nSearching for [blue]LoRA[/blue]'s...")
    LoRAs = findModels(userSettings["LoRAsLocation"],".safetensors")
    LoRAs.sort()

    print("\nSearching for [blue]ControlNet[/blue]'s...")
    controlNetWeights = findModels(userSettings["ControlNetsLocation"], ".pth")
    controlNetWeights.extend(findModels(userSettings["ControlNetsLocation"], ".safetensors"))
    controlNetWeights.extend(findModels(userSettings["ControlNetsLocation"], ""))
    controlNetWeights.append("Reference Only")
    controlNetWeights.sort()

    mainWeights = allWeights["safetensors"].copy()
    mainWeights.extend(allWeights["diffusers"].copy())
    mainWeights.extend(allWeights["huggingFace"].copy())
    mainWeights.extend(allWeights["ckpt"].copy())

    print(
        Panel(
            ":horse_racing: :horse_racing: :horse_racing: :horse_racing: It ain't much, but it's an honest day's work. Great work, cowgirls and cowboys! :horse_racing: :horse_racing: :horse_racing: :horse_racing: :horse_racing:",
            title = ":cowboy_hat_face:Weights Wrangled!:cowboy_hat_face:",
            box = box.ASCII,
            style = "yellow"
            )
        )

    return allWeights, mainWeights, VAEWeights, embeddingWeights, embeddingNames, controlNetWeights, LoRAs

# Find model/weights for program to use
def findModels(
        path,
        type,
        getHash = False
    ):
    if type == "":
        print("\nSearching for: [bold]Folder[/bold]")
        type = "Keras .h5"
    print("\nSearching for:[bold]", type)
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
            if "VAE" not in file and "embeddings" not in file and "controlnets" not in file and "." not in file and "diffusers" not in file and "tensorflow":
                # Skip 'VAE', 'embeddings', and hidden '.' folders
                print(" ",file," found!")
                currentList.append(file)
    
    print("...[green]finished![/green]")
    return currentList # returns the final list of files

def findImportedModel(dictionary, value):
    for category, weights in dictionary.items():
        #print("Category:",category)
        #print("Weights:",weights)
        for weight in weights:
            if str(weight) == str(value):
                #print("Searing for:",weight)
                #print("Found:",value)
                return category
    return None  # Value not found in the dictionary

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
    if whichModel == "VAE":
        thePatient = VAE
        filePath = userSettings["VAEModelsLocation"]
        dictionaryToFind = "state_dict"
        fileType = ".ckpt"
    elif whichModel == "Text Embeddings":
        thePatient = textEmbeddings
        filePath = userSettings["EmbeddingsLocation"]
        dictionaryToFind = "All"
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

    print("...done!")

    print("Saving analysis...")
    
    if readWriteFile.writeToFile(filePath + thePatient.replace(fileType,"-analysis.txt"), pytorchWeights, dictionaryToFind):
        print("...done!")
    
    pytorchWeights = None
    
    gc.collect()

def checkModel(selectedModel, legacy, dreamer):
    # load our main object/class

    dreamer.pytorchModel = selectedModel

    dreamer.legacy = legacy

    # Have we compiled any models already?
    if dreamer.generator is None:
        print("[yellow bold]\nNo Stable Diffusion model compiled![/yellow bold]\nThere must be a compiled model to analyze.\nCompiling now...",color.END)
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
    pytorchModel = None,
    legacyMode = None,
    typeOfModel = ".h5",
    dreamer = None,
    currentWeights = None,
    userSettings = None
):
    if dreamer == None:
        print("[bold red]Fatal Error![/bold red]:\n[yellow]The dreamer class was not passed into saveModel(). Unable to save!")
        return
    if currentWeights == None:
        print("[bold red]Fatal Error![/bold red]:\n[yellow]The modelsWeights were not passed into saveModel(). Unable to save!")
        return
    if userSettings == None:
        print("[bold red]Fatal Error![/bold red]:\n[yellow]The currentWeights, or userSettings were not passed into saveModel(). Unable to save!")
        return
    ## Determine model type
    if typeOfModel == "TensorFlow":
        typeOfModel = "tensorflow"
    elif typeOfModel == "Diffusers":
        typeOfModel = "diffusers"
    elif typeOfModel == "Safetensors":
        typeOfModel = "safetensors"
    else:
        typeOfModel = "safetensors"

    ## Make sure a model is built

    # Does dreamer have weights to use?
    if dreamer.pytorchModel is None:
        print("[yellow bold]\nNo Stable Diffusion model compiled/built![/yellow bold]\nThere must be a compiled/built model to save.\nCompiling/building now...")
        if pytorchModel is None:
            if userSettings["defaultModel"] != "":
                dreamer.pytorchModel = userSettings["defaultModel"]
            else:
                dreamer.pytorchModel = currentWeights[0]
        else:
            dreamer.pytorchModel = pytorchModel
        dreamer.legacy = legacyMode

    # Have we compiled/built any models already?
    if dreamer.generator is None or dreamer.generator.textEmbeddings is not None or dreamer.pytorchModel != pytorchModel:
        dreamer.pytorchModel = pytorchModel
        print("Compiling models without Text Embeddings")
        dreamer.compileDreams(embeddingChoices = None)
    
    ## Save Location
    # Load/create folder to save model to
    if typeOfModel != "safetensors":
        path = f"models/{typeOfModel}/{name}"
    else:
        path = f"models/safetensors/{name}"
    
    if not os.path.exists(path): #If it doesn't exist, create folder
        os.makedirs(path)


    # Set local variables
    model = dreamer.generator
    fileName = []

    # Save Model
    print("[cyan]\nSaving",name,"in the",typeOfModel,"format...")
    if typeOfModel == "tensorflow":
        for modelType in ["text_encoder", "diffusion_model", "decoder", "encoder"]:
            fileName.append(path + "/" + modelType + type)
        
        # Save Text Encoder
        print("\nSaving text encoder model as:\n",fileName[0])
        model.text_encoder.save(fileName[0])
        print("[green]Model saved![/green]")

        # Save Diffusion Model
        print("\nSaving diffusion model as:\n",fileName[1])
        model.diffusion_model.save(fileName[1])
        print("[green]Model saved![/green]")

        # Save Decoder
        print("\nSaving decoder model as:\n",fileName[2])
        model.decoder.save(fileName[2])
        print("[green]Model saved![/green]")

        # Save Encoder
        print("\nSaving encoder model as:\n",fileName[3])
        model.encoder.save(fileName[3])
        print("[green]Model saved![/green]")

        print("[green]Finished saving models![/green]")
    elif typeOfModel == "diffusers":
        dreamer.generator.pipeline.save_pretrained(
            save_directory = path,
            safe_serialization = True
        )
        print("[green]Finished saving model![/green]")