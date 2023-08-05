import cv2 #OpenCV
import os
import numpy as np
import tensorflow as tf
from .tileSetter import tileImage, setTiles

"""
Layers
"""

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

"""
Functions
"""

def preProcessControlNetImage(
        image,
        processingOption,
        imageSize = [512, 512],
        cannyOptions = [100, 200],
        tileScale = 4,
        upscaleMethod = "BICUBIC"
):
    """
    Pre-Process images for the model
    """

    ### Pre-Process the Image ###
    if processingOption == "Canny":
        print("\nPre-Processing image with Canny Detection...")
        print("Low Threshold:", cannyOptions[0])
        print("Low Threshold:", cannyOptions[1])
        detectedMap = cv2.Canny(image, cannyOptions[0], cannyOptions[1])
        print("...done!")
    elif processingOption == "HED":
        print("\nPre-Processing image with HED Detection...")
        detectedMap = HEDDetection(image)
        print("...done!")
    elif processingOption == "Tile":
        print("\nPre-Processing image with Tile Setter...")
        #print("Input Image Size:",image.shape)
        #print("Input Image kind:",type(image))
        print("Scale:",tileScale)
        print("Upscale Method:",upscaleMethod)
        detectedMap = tileImage(image, scale = tileScale, scaleMethod = upscaleMethod)
        print("...done!")
    elif processingOption == "BYPASS":
        detectedMap = image
        print("\nNo controlNet pre-processing...")
    elif processingOption == "None":
        detectedMap = image
        print("\nNo controlNet pre-processing...")

    if isinstance(detectedMap, list):
        #print("\nReceived tiles!")
        #print("Number of tiles:",len(detectedMap))
        control = []
        originalTileSize = detectedMap[0].shape[0]
        for item in detectedMap:
            #print("Tile shape:",item.shape)
            resizedItem = cv2.resize(item, (imageSize[0], imageSize[1]))
            #print("Reized tile:", resizedItem.shape)
            control.append(resizedItem)
        return control
    else:

        ### Match Image to Render Size ###
        detectedMap = cv2.resize(detectedMap, (imageSize[0], imageSize[1]))

        ### Prepare Image as Tensor ###
        detectedMap = HWC3(detectedMap)
        #control = tf.constant(detectedMap.copy(), dtype = tf.float32) / 255.0
        control = detectedMap

        return [control]


def previewProcessControlNetImage(
        image,
        processingOption,
        lowThreshold,
        highThreshold
):
    """
    Quick and Dirty preview of a specific process
    """
    if processingOption == "Canny":
        print("Previewing image with Canny Detection for preview")
        detectedMap = cv2.Canny(image, lowThreshold, highThreshold)
        print("...done!")
        return detectedMap
    
    if processingOption == "HED":
        print("Previewing image with HED Detection")
        detectedMap = HEDDetection(image)
        print("...done!")
        return detectedMap
    
def HEDDetection(image):
    """
    OpenCV Implementation of HED Detection
    """
    cv2.dnn_registerLayer('Crop', CropLayer)
    HEDModel = cv2.dnn.readNet("utilities/controlNetFiles/deploy.prototext", "utilities/controlNetFiles/hed_pretrained_bsds.caffemodel")

    input = cv2.dnn.blobFromImage(image, scalefactor = 1.0, size = (image.shape[0], image.shape[1]),
                                    mean = (104.00698793, 116.66876762, 122.67891434),
                                    swapRB = False, crop = False)
    HEDModel.setInput(input)
    output = HEDModel.forward()
    output = output[0, 0]
    output = cv2.resize(output, (image.shape[1], image.shape[0]))
    output = 255 * output
    output = output.astype(np.uint8)
    cv2.dnn_unregisterLayer('Crop')
    return output

def HWC3(x):
    """
    Height, Width, Chroma (aka, RGB, Red Green Blue).
    For example:(512, 512, 3)
    """
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis = 2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resizeImage(inputImage, resolution):
    H, W, C = inputImage.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(inputImage, (W, H), interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img