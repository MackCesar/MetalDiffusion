import cv2 as cv
import argparse
import numpy as np

"""
Holistically-Nested Edge Detection

Adapted by AJ Young from code written by (Ashutosh Chandra) and (Saining Xie at UC San Diego)

Original caffe model can be downloaded at: https://vcl.ucsd.edu/hed/5stage-vgg.caffemodel
The model will be need to be renamed to hed_pretrained_bsds.caffemodel

Runs purely through OpenCV, keeping it agnostic of any ML library like PyTorch or TensorFlow

"""

"""
Argument Parser
For individual use outside of dream.py
"""

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--prototxt', help='Path to deploy.prototxt',default='deploy.prototxt', required=False)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',default='hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

"""
Classes
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

class HEDDetector():
    def __init__(self):

        cv.dnn_registerLayer('Crop', CropLayer)

        # Load the model.
        self.HEDModel = cv.dnn.readNet("deploy.prototext", "hed_pretrained_bsds.caffemodel")
    
    def __call__(self,image):
        input = cv.dnn.blobFromImabe(image, scalefactor = 1.0, size = (512, 512),
                                     mean = (104.00698793, 116.66876762, 122.67891434),
                                     swapRB = False, crop = False)
        self.HEDModel.setInput(input)
        output = self.HEDModel.forward()
        output = output[0, 0]
        output = cv.resize(output, (image.shape[1], image.shape[0]))
        output = 255 * output
        # output = output.astype(np.uint8)
        # output = cv.cvtColor(output,cv.COLORGRAY2BGR)

        return output


"""while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    con=np.concatenate((frame,out),axis=1)
    if args.write_video:
        writer.write(np.uint8(con))
    cv.imshow(kWinName,con)"""