### System modules
import os
import warnings
import logging
import random
import sys

# Find model/weights for program to use
def findModels(path, type, getHash = False):
    print("\nSearching for models and weights...")
    directory = os.listdir(path.rstrip()) # local variable for an list of the files in the directory
    currentList = ["Stable Diffusion 1.4"] # local variable list for found files with a default already added
    for file in directory:
        if file.endswith(type):
            # Prints only type of file present in the folder
            print("... ",file," found!")
            if getHash is True:
                hash = " [" + modelHash(path + "/" + file) + "]"
            else:
                hash = ""
            finalName = str(file) + hash
            # print(finalName)
            currentList.append(finalName)
    
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