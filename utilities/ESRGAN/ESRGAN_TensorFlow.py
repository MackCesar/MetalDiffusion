import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

"""
ESRGAN via TensorFlow

Adapted by AJ Young from a tutorial by Google: https://www.tensorflow.org/hub/tutorials/image_enhancing

License: Creative Commons Attribution 4.0 and Apache 2.0 License

---

Upscales an image using Enhanced Super Resolution Generative Adversarial Network (ESRGAN) with TensorFlow

"""

def upscaleImage(imagePath = None, modelPath = "https://tfhub.dev/captain-pool/esrgan-tf2/1", saveResult = False):
    tf.print("\nUpscaling with ESRGAN")

    # Pre-process the image
    tf.print("\nPre-processing image...")
    preparedImage = preprocessImage(imagePath)
    tf.print("...done!")

    if saveResult is True:
        tf.print("\nSaving image...")
        saveImage(tf.squeeze(preparedImage),"OriginalImage")
        tf.print("...done!")
    
    # Download weights/model
    tf.print("\nLoading model...")
    model = hub.load(modelPath)
    tf.print("...done!")

    # Upscale image!
    tf.print("\nUpscaling image...")
    start = time.time()
    upscaledImage = model(preparedImage)
    upscaledImage = tf.squeeze(upscaledImage)
    tf.print("...done!\nTime Taken: %f" % (time.time() - start))

    if saveResult is True:
        saveImage(tf.squeeze(upscaledImage), "UpscaledImage")
    
    if not isinstance(upscaledImage, Image.Image):
        upscaledImage = tf.clip_by_value(upscaledImage, 0, 255)
        upscaledImage = Image.fromarray(tf.cast(upscaledImage, tf.uint8).numpy())

    return upscaledImage

def preprocessImage(image_path):
    """ Loads image from path and preprocesses to make it model ready
        Args:
        image_path: Path to the image file
    """
    if isinstance(image_path, str):
        hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    else:
        hr_image = image_path
    # If PNG, remove the alpha channel. The model only supports images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def saveImage(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("ESRGAN/%s.jpg" % filename)
    tf.print("\nSaved as %s.jpg" % filename)

if __name__ == "__main__":
    tf.print("\n\n\nWelcome to debug mode for ESRGAN-TensorFlow")

    finalImage = upscaleImage(
        imagePath = "ESRGAN/original.png",
        modelPath = "https://tfhub.dev/captain-pool/esrgan-tf2/1",
        saveResult = True
    )
    # Plotting Upscaled image
    # saveImage(tf.squeeze(finalImage), filename = "Fake Image")
