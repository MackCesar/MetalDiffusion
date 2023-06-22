import numpy as np
import math
import random

# Image modules
import cv2 #OpenCV
from PIL import Image, ImageDraw, ImageOps #Python Image Library

# Upscale Modules
try:
    from .ESRGAN.ESRGAN_TensorFlow import upscaleImage as ESRGANUpscaleTensorFlow
except Exception as e:
    print(e)

"""
Tile Setter

Created by AJ Young for MetalDiffusion:
https://github.com/soten355/MetalDiffusion

This program cuts an input image apart into smaller tiles. These tiles are then used for whatever reason;
in MetalDiffusion's case, upscaling via ControlNet Tile.

Once finished, the program can set the tiles back together into an image.
"""

def tileImage(
        image,
        scale = 2,
        scaleMethod = "BICUBIC",
        overlap = 32,
        name = None,
        debug = False
    ):
    ### Calculate the size of each sub-image including the overlap
    print("\nDesigning tiles...")
    ## Original Image Dimensions
    originalHeight = image.shape[0]
    originalWidth = image.shape[1]
    ## Tile Dimenions
    # Size of tile is determined by divisions(scale)
    tileSize = originalHeight // scale
    tileWidth = originalWidth // scale
    tileHeight = originalHeight // scale
    ## Calculate rows/columns of the "tiled" image
    totalRows = originalHeight // tileHeight
    totalColumns = originalWidth // tileWidth
    ## Apply scale to tile size:
    tileSize = tileSize * scale
    tileWidth = tileWidth * scale
    tileHeight = tileHeight * scale
    tileSizeWithOverlap = tileSize + overlap * 2

    ### Filename creator
    if name is None:
        # No file name given? Random name then
        randomNumber = np.random.randint(0, 9999)
        name = "tiledImage" + str(randomNumber)
    
    ### Scale Image
    ## Scale the image so the resulting tiles are scaled
    ## We're cutting tiles from the scaled image
    if scaleMethod == "ESRGAN":
        print("Upscaling with ESRGAN, TensorFlow")
        # ESRGAN Version
        image = ESRGANUpscaleTensorFlow(image, saveResult = debug)
        image = np.array(image)
    else:
        print("Upscaling with Bicubic")
        # Bicubic Verison
        image = Image.fromarray(image)
        image = image.resize( (originalWidth * scale, originalHeight * scale), resample = Image.BICUBIC)
        image = np.array(image)

    ### Create the tileSet variable that will be returned
    tileSet = []
    totalTiles = 0

    ### Debug Info
    if debug is True:
        print("Input Image Size:",originalWidth,originalHeight)
        print("Total Rows:", totalRows)
        print("Total Columns:", totalColumns)
        print("Scale:", scale)
        print("Tile Width before overlap:", tileWidth)
        print("Tile Height before overlap", tileHeight)
        print("Final Tile width: ",str(tileWidth + overlap * 2))
        print("Final Tile height: ",str(tileHeight + overlap * 2))
        print("Upscaled Image Size:",image.shape)
        print("Overlap:",overlap)
        print("Total Tiles:",totalRows*totalColumns)
    
    ### Loop through the rows and columns to get the tiles
    print("Cutting tiles...")
    for y in range(totalRows):
        
        # Find tile y position in image:
        imageY = y * tileHeight - 1
        if imageY < 0: imageY = 0

        for x in range(totalColumns):
            # Find tile x position in image:
            imageX = x * tileWidth - 1
            if imageX < 0: imageX = 0

            if debug is True: print("\nPosition: (",x,",",y,")")
            
            # Create/Reset Padding. We'll be mutliplying these numbers later

            if x == 0: #If we're in the first column
                padLeft = 0
            else:
                padLeft = 1
            if y == 0: #If we're in the first row
                padTop = 0
            else:
                padTop = 1
            if x == totalColumns - 1: # If we're in the last column
                padRight = 0
            else:
                padRight = 1
            if y == totalRows - 1: # If we're in the last row
                padBottom = 0
            else:
                padBottom = 1

            # Calculate the required padding for each side
            # If we're at an edge of the image, then the variable will be 0
            padLeft = padLeft * (-1 * overlap)
            padRight = padRight * (overlap)
            padTop = padTop * (-1 * overlap)
            padBottom = padBottom * (overlap)

            # Final padding corrections

            if padLeft == 0: padRight += overlap
            if padRight == 0: padLeft -= overlap
            if padTop == 0: padBottom += overlap
            if padBottom == 0: padTop -= overlap

            if debug is True: print("Padding left, right, top, bottom:",padLeft, padRight, padTop, padBottom)

            # Cut Tile
            if debug is True:
                print("Left: ",imageX+padLeft)
                print("Right: ",imageX+padRight+tileSize)
                print("Top: ",imageY+padTop)
                print("Bottom: ",imageY+padBottom+tileSize)
            #tile = image[imageX+padLeft:imageX+padRight+tileWidth, imageY+padTop:imageY+padBottom+tileHeight]
            tile = image[imageY+padTop:imageY+padBottom+tileHeight, imageX+padLeft:imageX+padRight+tileWidth]

            if debug is True: print("Final Shape After Cut:",tile.shape)

            tile = Image.fromarray(tile)
            tile = tile.resize( (tileWidth, tileHeight), resample = Image.BICUBIC)
            tile = np.array(tile)

            if debug is True: print("Final Tile after Scale:", tile.shape)

            # Add Tile to collection
            tileSet.append(tile)
            totalTiles += 1

            if debug is True:

                # Save the individual tiles
                cv2.imwrite(f'tileTest/tile{imageX}_{imageY}_{totalTiles}.png', tile)

    print("Finished! Returning tiles...")
    return tileSet

def setTiles(
        tileSet,
        overlap = 32,
        scale = 2,
        name = None,
        debug = False
    ):
    print("\nSetting Tiles...")
    print("...configuring tile layout...")

    ### Tile dimensions ###
    totalTiles = len(tileSet) # aka, divisions
    scale = int(math.sqrt(totalTiles))
    # Input tile size, we're assuming it's been downscaled from tileImage
    inputTileSize = tileSet[0].shape[0]
    inputTileWidth = tileSet[0].shape[1]
    inputTileHeight = tileSet[0].shape[0]
    # Final Tile Size
    tileSize = inputTileSize + 2 * overlap
    tileWidth = inputTileWidth + overlap * 2
    tileHeight = inputTileHeight + overlap * 2

    ### Image Dimensions ###
    totalRows = int(math.sqrt(totalTiles))
    totalColumns = totalRows
    finalWidth = totalColumns * inputTileWidth
    finalHeight = totalRows * inputTileHeight

    # Create final image
    overlayingImage = Image.new("RGBA", (finalWidth, finalHeight))
    finalImage = Image.new("RGBA", (finalWidth, finalHeight), color = None)

    # Create gradient for tile overlap
    gradient = Image.linear_gradient("L")

    ### Debug Info ###
    if debug is True:
        print("Total Tiles:",totalTiles)
        print("Input Tile Size:", inputTileWidth,inputTileHeight)   
        print("Tile Size with Overlap:",tileWidth, tileHeight)
        print("Scale:", scale)
        print("Overlap:", overlap)
        print("Total Rows:",totalRows)
        print("Total Columns:",totalColumns)
        print("\nFinal Image Shape:",finalImage.size)

    if name is None:
        # No file name given? Random name then
        randomNumber = np.random.randint(0, 9999)
        name = "tiledImage" + str(randomNumber)
    
    tileCount = 0

    print("...placing tiles...")

    for y in range(totalRows):

        # Calculate the vertical position
        imageY = y * inputTileHeight - 1
        if imageY < 0: imageY = 0
        if y != 0:
            imageY -= overlap
            if y == totalRows - 1:
                imageY -= overlap

        for x in range(totalColumns):

            # Calculate the horizontal position
            imageX = x * inputTileWidth - 1
            if imageX < 0: imageX = 0
            if x != 0:
                imageX -= overlap
                if x == totalColumns - 1:
                    imageX -= overlap

            # if debug is True: print("\nPosition before adjustment:\n(",imageX,",",imageY,")")

            # Final position adjustments

            tile = tileSet[tileCount]
            if debug is True: print("\nTile Shape:",tile.shape)

            tile = Image.fromarray(tile)
            tile = tile.convert("RGBA") # Add alpha channel
            tile = tile.resize( (tileWidth, tileHeight), resample = Image.BICUBIC) # Restore input tiles to original size

            if debug is True: print("Resized Tile:",tile.size)

            if debug is True: print("Position:\n(",imageX,",",imageY,")")
            
            ## Create the gradient for overlap
            finalGradient = Image.new("L", (tileWidth, tileHeight), color = "white")

            # Add individual gradients based on tile position

            if y != 0: # Top Gradient
                if y == totalRows - 1 or y == 1: # If we're at a border or touching a border tile
                    finalGradient.paste(gradient.resize( (tileWidth, overlap * 2 + overlap), resample = Image.BICUBIC), (0, 0 - overlap) )
                else:
                    finalGradient.paste(gradient.resize( (tileWidth, overlap + overlap // 2), resample = Image.BICUBIC), (0, 0 - overlap // 2) )
            
            if y != totalRows - 1: # Bottom Gradient
                if y == 0 or y == totalRows - 2: # If we're at a border
                    finalGradient.paste(gradient.rotate(180).resize( (tileWidth, overlap * 2 + overlap), resample = Image.BICUBIC), (0, tileHeight - (overlap * 2) + overlap))
                else:
                    finalGradient.paste(gradient.rotate(180).resize( (tileWidth, overlap + overlap // 2), resample = Image.BICUBIC), (0, tileHeight - (overlap) + overlap // 2))
            
            if x != 0: # Left Gradient
                if x == totalColumns -1 or x == 1: # If we're at a border
                    finalGradient.paste(gradient.rotate(90).resize( (overlap * 2 + overlap, tileHeight), resample = Image.BICUBIC), (0 - overlap, 0) )
                else:
                    finalGradient.paste(gradient.rotate(90).resize( (overlap + overlap // 2, tileHeight), resample = Image.BICUBIC), (0 - overlap // 2, 0) )
            
            if x != totalColumns - 1: # Right Gradient
                if x == 0 or x == totalColumns - 2: # If we're at a border
                    finalGradient.paste(gradient.rotate(270).resize( (overlap * 2 + overlap, tileHeight), resample = Image.BICUBIC), (tileWidth - (overlap * 2) + overlap, 0) )
                else:
                    finalGradient.paste(gradient.rotate(270).resize( (overlap + overlap // 2, tileHeight), resample = Image.BICUBIC), (tileWidth - overlap + overlap // 2, 0) )
            
            if debug is True: finalGradient.save(f"tileTest/{x}_{y}_finalGradient.png")
            

            if debug is True: print("Now applying:")

            if debug is True:
                tileBeforePaste = Image.new("RGBA", (tileWidth, tileHeight), "black")
                tileBeforePaste.paste(tile, (0, 0), mask = finalGradient)
                tileBeforePaste.save(f"tileTest/tileBeforePaste_{x}_{y}.png")
            
            # Place the tile in two ways:
            #overlayingImage.paste(tile, (imageX, imageY), mask = finalGradient) # With alpha's
            finalImage.paste(tile, (imageX, imageY), mask = finalGradient) # as the underlying image

            if debug is True: finalImage.save(f"tileTest/tile_{x}_{y}.png")

            tileCount += 1
    
    #finalImage = Image.alpha_composite(finalImage, overlayingImage)
    #finalImage = Image.blend(overlayingImage,finalImage, 0.1)
    
    if debug is True: finalImage.save("tileTest/finalCombinedImage.png")
    return finalImage


"""
This section is for running the script on it's own
"""

if __name__ == "__main__":
    image = cv2.imread("tileTest/tileTest005.png", cv2.IMREAD_COLOR)
    print("\nInput image shape:",image.shape)

    print("\nImage loaded. Tiling...")
    # Split the image into four 256x256 sub-images
    tiledImage = tileImage(image, scale = 8, debug = False)
    print("\n...tiled!")


    # Display the tiles
    #print("Displaying tiles:")

    originalResolution = tiledImage[0].shape[0]
    print("\nOriginal Resolution:",originalResolution)
    
    print("Resizing tiles to 512x...")
    for i in range(len(tiledImage)):
        #print("\nOld Tile Shape:", tiledImage[i].shape)
        tempTile = cv2.cvtColor(tiledImage[i], cv2.COLOR_BGR2RGB)
        tempTile = Image.fromarray(tempTile)
        tempTile = tempTile.resize((512,768),resample = Image.BICUBIC)
        #print(tempTile.size)
        tiledImage[i] = np.array(tempTile)
        #print("New Tile Shape:", tiledImage[i].shape)

    finalImage = setTiles(tiledImage, debug = False)

    print("Displaying Image:")

    print("Final Image Size:", finalImage.size)

    #finalImage = cv2.imread(finalImage, cv2.IMREAD_COLOR)

    #cv2.imshow('Final Image', finalImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    print("Good bye!\a\a")