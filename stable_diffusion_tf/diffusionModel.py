import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Zeros

from .layers import apply_seq, td_dot, GEGLU

### Main Model
class UNetModel(keras.models.Model):
    def __init__(self):
        super().__init__()

        #SD 2.0 Config

        imageSize = 32
        inChannels = 4 # Channels for the input tensor
        modelChannels = 320 # Base channel count for model
        outChannels = 4 # Channels for the output tensor
        numberResBlocks = 2 # Number of residual blocks per down sample
        attentionResolutions = [4, 2, 1] # a collection of downsample rates at which attention will take place. May be a set, list, or tuple.
            #For example, if this contains 4, then at 4x downsampling, attention will be sued
        dropout = 0 # Dropout probability
        channelMultiplier = [1, 2, 4, 4] # Channel multiplyer for each level of the UNet
        convolutionResample = True # if True, use learned convolutions for upsampling and downsampling.
        dimensions = 2 # Dimensions of convolutional layer
        numberClasses = None
        useCheckpoint = False
        use_fp16 = False
        numberHeads = -1 # Number of Heads
        numberHeadChannels = 64 # Number of channel per head
        numberHeadsUpsample = -1 # Number of heads upsample
        timeEmbedDimensions = modelChannels * 4

        # Variable Adjustments

        if numberHeadsUpsample == -1:
            numberHeadsUpsample = numberHeads
        
        if numberHeads == -1:
            assert numberHeadChannels != -1, "Either number of heads or number of channels per head need to be set"
        
        if numberHeadChannels == -1:
            assert numberHeads != -1, "Either number of heads or number of channels per head need to be set"
        
        self.timeEmbed = [
            keras.layers.Dense(timeEmbedDimensions, input_shape = (modelChannels,), name = "Time_Embed01"),
            keras.activations.swish,
            keras.layers.Dense(1280, input_shape = (timeEmbedDimensions,), name = "Time_Embed02"),
        ]

        # Input Blocks
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size = 3, padding = 1)],
            [ResBlock(320, 320), SpatialTransformer(320, 5, 64)],
            [ResBlock(320, 320), SpatialTransformer(320, 5, 64)],
            [Downsample(320)],
            [ResBlock(320, 640), SpatialTransformer(640, 10, 64)],
            [ResBlock(640, 640), SpatialTransformer(640, 10, 64)],
            [Downsample(640)],
            [ResBlock(640, 1280), SpatialTransformer(1280, 20, 64)],
            [ResBlock(1280, 1280), SpatialTransformer(1280, 20, 64)],
            [Downsample(1280)],
            [ResBlock(1280, 1280)],
            [ResBlock(1280, 1280)],
        ]
        self.middle_block = [
            ResBlock(1280, 1280),
            SpatialTransformer(1280, 20, 64),
            ResBlock(1280, 1280),
        ]
        self.output_blocks = [
            [ResBlock(2560, 1280)],
            [ResBlock(2560, 1280)],
            [ResBlock(2560, 1280), Upsample(1280)],
            [ResBlock(2560, 1280), SpatialTransformer(1280, 20, 64)],
            [ResBlock(2560, 1280), SpatialTransformer(1280, 20, 64)],
            [
                ResBlock(1920, 1280),
                SpatialTransformer(1280, 20, 64),
                Upsample(1280),
            ],
            [ResBlock(1920, 640), SpatialTransformer(640, 10, 64)],
            [ResBlock(1280, 640), SpatialTransformer(640, 10, 64)],
            [
                ResBlock(960, 640),
                SpatialTransformer(640, 10, 64),
                Upsample(640),
            ],
            [ResBlock(960, 320), SpatialTransformer(320, 5, 64)],
            [ResBlock(640, 320), SpatialTransformer(320, 5, 64)],
            [ResBlock(640, 320), SpatialTransformer(320, 5, 64)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon = 1e-5, name = "Out"),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size = 3, padding = 1, zero = True),
        ]

    def call(self, inputs):
        x, t_emb, context = inputs
        emb = apply_seq(t_emb, self.timeEmbed)

        def apply(x, layer): # TimestepEmbedSequential from OpenAI Model on Stability AI
            if isinstance(layer, ResBlock): # Resblock is TimestepBlock from OpenAI Model on Stability AI
                x = layer([x, emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)

        for layer in self.middle_block:
            x = apply(x, layer)

        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                x = apply(x, layer)
        return apply_seq(x, self.out)

class PaddedConv2D(keras.layers.Layer):
    def __init__(
        self,
        channels,
        kernel_size,
        padding = 0,
        stride = 1,
        name = None,
        zero = False
    ):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D((padding, padding), name = "Padding2D")
        if zero is False:
            self.conv2d = keras.layers.Conv2D(
                channels, kernel_size, strides=(stride, stride), name = name
            )
        else:
            self.conv2d = keras.layers.Conv2D(
                channels,
                kernel_size,
                strides = (stride, stride),
                name = name,
                kernel_initializer = Zeros(),
                bias_initializer = Zeros()
            )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)

class ResBlock(keras.layers.Layer): # Residual Block
    def __init__(
        self,
        channels,
        out_channels
    ):
        super().__init__()
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon = 1e-5, name = "ResBlock_In"),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding = 1, name = "in_layers"),
        ]
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(out_channels, name = "ResBlock_Embedded"),
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon = 1e-5, name = "ResBlock_Out"),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding = 1, name = "out_layers"),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_head):
        super().__init__()
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False, name = "ToQuery") # To Query
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False, name = "ToKey") # To Key
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False, name = "ToValue") # To Value
        self.scale = d_head**-0.5
        self.numberHeads = n_heads
        self.head_size = d_head
        self.to_out = [keras.layers.Dense(n_heads * d_head, name = "ToOut")] # To Out

    def call(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context) # Query, Key, Value
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.numberHeads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.numberHeads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.numberHeads, self.head_size))

        q = keras.layers.Permute((2, 1, 3))(q)  # (bs, numberHeads, time, head_size)
        k = keras.layers.Permute((2, 3, 1))(k)  # (bs, numberHeads, head_size, time)
        v = keras.layers.Permute((2, 1, 3))(v)  # (bs, numberHeads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, numberHeads, time, time)
        attention = td_dot(weights, v)
        attention = keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, numberHeads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.numberHeads * self.head_size))
        return apply_seq(h_, self.to_out)


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5, name = "BasicTransformerBlock_Norm1")
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5, name = "BasicTransformerBlock_Norm2")
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5, name = "BasicTransformerBlock_Norm3")
        self.geglu = GEGLU(dim * 4, name = "FF")
        self.dense = keras.layers.Dense(dim, name = "FF")

    def call(self, inputs):
        x, context = inputs
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class SpatialTransformer(keras.layers.Layer):
    # Transformer for image like data
    def __init__(
        self,
        channels, # Number of filter channels
        n_heads, # Number of Heads
        d_head # Dimension of Head
    ):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5, name = "SpatialTransformerNormalization")
        assert channels == n_heads * d_head
        inChannels = channels
        innerDimensions = n_heads * d_head
        self.proj_in = keras.layers.Dense(
            inChannels,
            input_shape = (innerDimensions,),
            name = "proj_in"
        )
        self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head)]
        self.proj_out = keras.layers.Dense(
            inChannels,
            input_shape = (innerDimensions,),
            name = "proj_out",
            kernel_initializer = Zeros(),
            bias_initializer = Zeros()
        )

    def call(self, inputs):
        x, context = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + x_in


class Downsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.op = PaddedConv2D(channels, 3, stride = 2, padding = 1)

    def call(self, x):
        return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size = (2, 2))
        self.conv = PaddedConv2D(channels, 3, padding = 1)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)
