"""
3d Rotation via Python OpenCV by https://gist.github.com/flufy3d/7580a8a18356d5527bbc
"""

import cv2
import numpy as np
import time
from numpy import *
import math

### Console GUI
from rich import print, box
from rich.panel import Panel
from rich.text import Text

def rotateImage(
        inputImage,
        dx,
        dy,
        dz,
        alpha,
        beta,
        gamma,
        focalLength
    ):

    print(":robot: Image transformer: OpenCV in disguise! :robot:")

    if isinstance(inputImage, np.ndarray) is False:
        print("...received PIL Image, converting to NumPy...")
        inputImage = np.array(inputImage)

    alpha = (alpha - 90.) * math.pi / 180.
    beta = (beta - 90.) * math.pi / 180.
    gamma = (gamma - 90.) * math.pi / 180.

    # Get width and height for ease of use in matrices
    rows,cols = inputImage.shape[:2]
    w = cols
    h = rows

    # Projection 2D -> 3D matrix
    A1 = np.array([
                  [1, 0, -w/2],
                  [0, 1, -h/2],
                  [0, 0,    0],
                  [0, 0,    1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([
              [1,          0,           0, 0],
              [0, math.cos(alpha), -math.sin(alpha), 0],
              [0, math.sin(alpha),  math.cos(alpha), 0],
              [0,          0,           0, 1]])
    RY = np.array([
              [math.cos(beta), 0, -math.sin(beta), 0],
              [0, 1,          0, 0        ],
              [math.sin(beta), 0,  math.cos(beta), 0],
              [0, 0,          0, 1]       ])
    RZ = np.array([
              [math.cos(gamma), -math.sin(gamma), 0, 0],
              [math.sin(gamma),  math.cos(gamma), 0, 0],
              [0,          0,           1, 0],
              [0,          0,           0, 1]])
    
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX ,RY) , RZ)

    # Translation matrix
    T = np.array([
             [1, 0, 0, dx],
             [0, 1, 0, dy],
             [0, 0, 1, dz],
             [0, 0, 0, 1]])
    
    # 3D -> 2D matrix
    A2 = np.array([
              [focalLength, 0, w/2, 0],
              [0, focalLength, h/2, 0],
              [0, 0,   1, 0]])
    
    # Final transformation matrix
    finalTransformation = np.dot(A2,np.dot(T,np.dot(R,A1)))

    print("[bold]Transforming![/bold]")

    # Apply matrix transformation
    outputImage = cv2.warpPerspective(
        inputImage,
        finalTransformation,
        (w,h),
        borderMode = cv2.BORDER_REFLECT
    )

    return outputImage

if __name__ == "__main__":

    print("Okay, let's see if we can rotate some images in 3 dimenions")

    ## Create an image
    print("Loading images...")
    img_list = [cv2.imread("tileTest/tileTest007.png", cv2.IMREAD_COLOR), cv2.imread("tileTest/tileTest002.png", cv2.IMREAD_COLOR)]

    ## Create a window to rotate the image
    cv2.namedWindow('cmd' )
    cv2.resizeWindow('cmd', 700, 625)

    ## Rotation and translation values
    rot_x = 90
    rot_y = 90
    rot_z = 90
    trans_x = 0
    trans_y = 0
    trans_z = 200
    focal = 200
    bg_weight = 1.0
    fg_weight = 1.0

    ## When the user changes a value on the GUI
    def on_change(value,type): 
        if type.find('weight') == -1:
            globals()[type] = value
        else:
            globals()[type] = (value/100.)

    ## Create GUI objects
    cv2.createTrackbar('rot x', 'cmd', 90, 360, lambda value:on_change(value,'rot_x'))
    cv2.createTrackbar('rot y', 'cmd', 90, 360, lambda value:on_change(value,'rot_y'))
    cv2.createTrackbar('rot z', 'cmd', 90, 360, lambda value:on_change(value,'rot_z'))
    cv2.createTrackbar('trans x', 'cmd', 0, 1000, lambda value:on_change(value,'trans_x'))
    cv2.createTrackbar('trans y', 'cmd', 0, 1000, lambda value:on_change(value,'trans_y'))
    cv2.createTrackbar('trans z', 'cmd', 200, 1000, lambda value:on_change(value,'trans_z'))
    cv2.createTrackbar('focal', 'cmd', 200, 1000, lambda value:on_change(value,'focal'))
    cv2.createTrackbar('fg_weight', 'cmd', 100, 100, lambda value:on_change(value,'fg_weight'))
    cv2.createTrackbar('bg_weight', 'cmd', 100, 100, lambda value:on_change(value,'bg_weight'))

    ## Loop variables

    while True:

        bg = np.array(np.random.randint(255, size = (512,512)), dtype = np.uint8)

        img = rotateImage(img_list[0], trans_x, trans_y, trans_z, rot_x,rot_y, rot_z, focal);
        cv2.imshow('img',img)

        if 0xFF & cv2.waitKey(1) == 27:
            print("Exiting...")
            break

        time.sleep(0.02)