import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from .layers import apply_seq
from .kerasCVDiffusionModels import GroupNormalization, PaddedConv2D 

class Decoder(keras.Sequential):
    def __init__(
        self,
        img_height,
        img_width,
        name = None,
        download_weights = False):
        super().__init__(
            [
                keras.layers.Input((img_height // 8, img_width // 8, 4)),
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(4, 1, name = "PostQuantConvolutionalIn"),
                PaddedConv2D(512, 3, padding = 1, name = "ConvolutionalIn"),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(size = (2,2)),
                PaddedConv2D(512, 3, padding = 1),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(size = (2,2)),
                PaddedConv2D(512, 3, padding = 1),
                ResnetBlock(256),
                ResnetBlock(256),
                ResnetBlock(256),
                keras.layers.UpSampling2D(size = (2,2)),
                PaddedConv2D(256, 3, padding = 1),
                ResnetBlock(128),
                ResnetBlock(128),
                ResnetBlock(128),
                GroupNormalization(epsilon = 1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding = 1, name = "ConvolutionalOut"),
            ],
            name=name,
        )

        if download_weights:
            decoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
                file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
            )
            self.load_weights(decoder_weights_fpath)

class ImageEncoder(keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""
    
    def __init__(
            self,
            img_height = 512,
            img_width = 512,
            download_weights = False
        ):
        super().__init__(
            [
                keras.layers.Input((img_height, img_width, 3)),
                PaddedConv2D(128, 3, padding = 1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding = 1, strides = 2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding = 1, strides = 2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding = 1, strides = 2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                GroupNormalization(epsilon = 1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding = 1),
                PaddedConv2D(8, 1),
                # TODO(lukewood): can this be refactored to be a Rescaling layer?
                # Perhaps some sort of rescale and gather?
                # Either way, we may need a lambda to gather the first 4 dimensions.
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )

"""
Blocks
"""

class ResnetBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
        })
        return config

class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
        })
        return config

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        _, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * (c**-0.5)
        y = keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs