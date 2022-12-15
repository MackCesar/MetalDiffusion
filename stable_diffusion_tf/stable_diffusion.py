### System modules
from tqdm import tqdm #progress bar module
import sys
import os
import warnings
import logging

### Math modules
import numpy as np
import math
import random

### Time modules
import datetime

### Memmory Management
import gc #Garbage Collector

### Import tensorflow module
### but with supressed warnings to clear up the terminal outputs
# Filter tensorflow version warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

# tensorflow optimizations
#tf.config.optimizer.set_jit(True)

### Keras module
from tensorflow import keras
try:
   from keras import backend as K
except Exception as e:
   print(e)

### Modules for Machine Learning
## VAE, encode and decode
from .legacyAutoencoderKl import Decoder, Encoder
from .autoencoderKl import Decoder, Encoder
## Diffusion
from .diffusionModel import UNetModel
from .legacyDiffusionModel import LegacyUNetModel
## Text encoder
# Legacy, older than SD 2.0
from .clip_encoder import CLIPTextTransformer
# SD 2.0 and newer
from .openClipEncoder import OpenCLIPTextTransformer
# Tokenizer
from .clip_tokenizer import SimpleTokenizer, LegacySimpleTokenizer
# Pytorch (for converting pytorch weights)
import torch as torch

### Modules for image building
from PIL import Image

### Global Variables
MAX_TEXT_LEN = 77
from .constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD, PYTORCH_CKPT_MAPPING

### Main Class

class StableDiffusion:
    def __init__(
        self,
        img_height = 512,
        img_width = 512,
        jit_compile = False,
        download_weights = True,
        pyTorchWeights = None,
        legacy = True
    ):
        # Set image dimensions
        self.img_height = img_height
        self.img_width = img_width

        # Set tokenizer
        if legacy is True:
            self.tokenizer = LegacySimpleTokenizer()
            self.legacy = True
        else:
            self.tokenizer = SimpleTokenizer()
            self.legacy = False

        ### Weights and models
        # Check for pytorch weights, if so we will load them later
        # If not, then we're loading in a pre-compiled model
        if ".ckpt" in pyTorchWeights:
            self.pyTorchWeights = pyTorchWeights
            pyTorchWeights = None
        
        # Load/compile models
        text_encoder, diffusion_model, decoder, encoder = get_models(
            img_height,
            img_width,
            download_weights = download_weights,
            preCompiled = pyTorchWeights, # If not None, then we're passing on the model to be loaded,
            legacy = legacy
        )
        
        # Create object variables that point to the models
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.encoder = encoder

        # Load pytorch weights if pytorch weights were given
        # must be done after compiling models
        if pyTorchWeights is None:
            self.load_weights_from_pytorch_ckpt(self.pyTorchWeights, legacy)

        # Just in time compilation
        if jit_compile:
            self.text_encoder.compile(jit_compile = True)
            self.diffusion_model.compile(jit_compile = True)
            self.decoder.compile(jit_compile = True)
            self.encoder.compile(jit_compile = True)
        else:
            self.text_encoder.compile(optimizer = "adam")
            self.diffusion_model.compile(optimizer = "adam")
            self.decoder.compile(optimizer = "adam")
            self.encoder.compile(optimizer = "adam")
        
        # Global policy
        self.dtype = tf.float32 # Default

        # Maaaybe float16 will result in faster images? Switch to float16 if global policy says so
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.dtype = tf.float16

        ## Prompts
        self.prompt = None
        self.negativePrompt = None
        # Cache
        self.encodedPrompt = None
        self.encodedNegativePrompt = None
        self.batch_size = None

    # Generate an image
    def generate(
        self,
        prompt,
        negativePrompt = None,
        batch_size = 1,
        num_steps = 25,
        unconditional_guidance_scale = 7.5,
        temperature = 1,
        seed = None,
        input_image = None,
        input_image_strength = 0.5,
        input_mask = None
    ):
        ## Memory Efficiency
        # Clear up tensorflow memory
        print("\n...cleaning memory...")
        tf.keras.backend.clear_session()
        gc.collect()

        # Use the generator function within the newly created class to generate an array that will become an image
        print("...getting to work...")

        # Prompts
        if self.prompt != prompt: # New prompt?
            self.prompt = prompt
            self.encodedPrompt = None
        
        if self.negativePrompt != negativePrompt: # New negative prompt?
            self.negativePrompt = negativePrompt
            self.encodedNegativePrompt = None
        
        if self.batch_size != batch_size: # clear cache is batch_size has changed
            self.encodedPrompt = None
            self.encodedNegativePrompt = None
        
        # Tokenize prompt (i.e. starting context)

        # Encode prompt tokens (and their positions) into a "context vector"
        if self.encodedPrompt is None or self.encodedNegativePrompt is None:
            print("\n...tokenizing prompt...")
            
            if self.legacy is True:
                ## Legacy mode

                phrase, pos_ids = self.legacyPreparePrompt(prompt, batch_size)

                context = self.text_encoder([phrase, pos_ids], training = False)
                self.encodedPrompt = context
            else:
                ## Contemporary mode

                phrase, pos_ids = self.preparePrompt(prompt, batch_size)
                context = self.text_encoder([phrase, pos_ids], training = False)
                self.encodedPrompt = context #Store context as concoded prompt, basically a cache
        else:
            print("...using cached encoded prompt...")
            context = self.encodedPrompt

        # Prepare the input image, if it was given
        input_image_tensor = None
        if input_image is not None:
            print("...preparing input image...")
            if type(input_image) is str:
                input_image = Image.open(input_image)
                input_image = input_image.resize((self.img_width, self.img_height))

            elif type(input_image) is np.ndarray:
                input_image = np.resize(input_image, (self.img_height, self.img_width, input_image.shape[2]))
                
            input_image_array = np.array(input_image, dtype=np.float32)[None,...,:3]
            input_image_tensor = tf.cast((input_image_array / 255.0) * 2 - 1, self.dtype)
        
        # Prepare the image mask, if it was given
        if type(input_mask) is str:
            input_mask = Image.open(input_mask)
            input_mask = input_mask.resize((self.img_width, self.img_height))
            input_mask_array = np.array(input_mask, dtype = np.float32)[None,...,None]
            input_mask_array =  input_mask_array / 255.0
            
            latent_mask = input_mask.resize((self.img_width//8, self.img_height//8))
            latent_mask = np.array(latent_mask, dtype = np.float32)[None,...,None]
            latent_mask = 1 - (latent_mask.astype("float") / 255.0)
            latent_mask_tensor = tf.cast(tf.repeat(latent_mask, batch_size , axis = 0), self.dtype)

        # Create a random seed if one is not provided
        if seed is None:
            print("...generating random seed...")
            seed = random.randint(1000, sys.maxsize)

        # Tokenize negative prompt or use default padding tokens
        if self.encodedNegativePrompt is None:
            print("...tokenizing negative prompt...")
            
            if self.legacy is True:
                if negativePrompt is None:
                    negativePrompt = ""
                unconditional_tokens, pos_ids = self.legacyPreparePrompt(negativePrompt, batch_size)
            else:
                if negativePrompt is None:
                    negativePrompt = ""
                unconditional_tokens, pos_ids = self.preparePrompt(negativePrompt, batch_size)
            
            unconditional_context = self.text_encoder(
                [unconditional_tokens, pos_ids],
                training = False
            )
            self.encodedNegativePrompt = unconditional_context
        else:
            print("...using cached encoded negative prompt...")
            unconditional_context = self.encodedNegativePrompt

        # Establish time steps
        print("...establishing time steps...")
        timesteps = np.arange(1, 1000, 1000 // num_steps)

        # Input image time steps
        print("...establishing input image time steps...")
        input_img_noise_t = timesteps[ int(len(timesteps)*input_image_strength) ]

        # Get starting parameters for generation
        print("...loading starting parameters...")
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps,
            batch_size,
            seed,
            input_image = input_image_tensor,
            input_img_noise_t = input_img_noise_t
        )

        # Final timesteps calculation if an input image is given
        print("...finalizing time steps...")
        if input_image is not None:
            timesteps = timesteps[: int(len(timesteps)*input_image_strength)]

        # Diffusion stage
        print("...starting diffusion...\n")
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar: #Iterate the samples through the progress bar
            progbar.set_description(f"{index:3d} {timestep:3d}") # Update Progress Bar label
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
                batch_size,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature, seed
            )

            if input_mask is not None and input_image is not None:
                # If mask is provided, noise at current timestep will be added to input image.
                # The intermediate latent will be merged with input latent.
                latent_orgin, alphas, alphas_prev = self.get_starting_parameters(
                    timesteps,
                    batch_size,
                    seed,
                    input_image = input_image_tensor,
                    input_img_noise_t = timestep
                )
                latent = latent_orgin * latent_mask_tensor + latent * (1- latent_mask_tensor)

        # Decoding stage
        print("\n...decoding diffusion...")
        decoded = self.decoder(latent, training = False)
        decoded = ((decoded + 1) / 2) * 255

        # Merging of inpainting result of input mask with original image

        if input_mask is not None:
          # Merge inpainting output with original image
          decoded = input_image_array * (1-input_mask_array) + np.array(decoded) * input_mask_array
        
        #Memory cleanup
        gc.collect()

        # return final image as an array
        return np.clip(decoded, 0, 255).astype("uint8")

    def timestep_embedding(
        self,
        timesteps,
        dim = 320,
        max_period = 10000
    ):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))


    def add_noise(self, x , t , noise = None ):
        batch_size,w,h = x.shape[0] , x.shape[1] , x.shape[2]
        if noise is None:
            noise = tf.random.normal((batch_size,w,h,4), dtype=self.dtype)
        sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return  sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def get_starting_parameters(
        self,
        timesteps,
        batch_size,
        seed, 
        input_image = None,
        input_img_noise_t = None
    ):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if input_image is None:
            # Create a random input image
            latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        else:
            # Encode the given image
            latent = self.encoder(input_image, training = False)
            # Repeat it for the given batch size
            latent = tf.repeat(latent , batch_size , axis = 0)
            # Noise the image
            latent = self.add_noise(latent, input_img_noise_t)
        return latent, alphas, alphas_prev

    def get_model_output(
        self,
        latent,
        inputTimesteps,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
    ):

        # Establish timestep embedding
        timesteps = np.array([inputTimesteps])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis = 0)

        # Get unconditional (negative prompt) latent image
        unconditional_latent = self.diffusion_model(
            [latent, t_emb, unconditional_context],
            training = False
        )
        # Get conditional (positive prompt) latent image
        latent = self.diffusion_model(
            [latent, t_emb, context],
            training = False
        )

        # Combine the images and return the result
        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0
    
    # Load pytorch weights as models
    def load_weights_from_pytorch_ckpt(
        self,
        pytorch_ckpt_path,
        legacy = True
    ):
        print("\nLoading pytorch checkpoint " + pytorch_ckpt_path)
        pytorchWeights = torch.load(pytorch_ckpt_path, map_location = "cpu")
        if legacy is True:
            ## Legacy Mode
            print("...loading pytroch weights in legacy mode...")
            for module_name in ['text_encoder_legacy', 'diffusion_model', 'decoder', 'encoder' ]:
                module_weights = []
                for i , (key , perm ) in enumerate(PYTORCH_CKPT_MAPPING[module_name]):
                    weight = pytorchWeights['state_dict'][key].numpy()
                    if perm is not None:
                        weight = np.transpose(weight , perm )
                    module_weights.append(weight)
                if module_name == "text_encoder_legacy":
                    module_name = "text_encoder"
                getattr(self, module_name).set_weights(module_weights)
                print("Loaded %d pytorch weights for %s"%(len(module_weights) , module_name))
        else:
            ## Contemporary Mode
            print("...loading pytorch weights in contemporary mode...")
            for module_name in ['text_encoder', 'diffusion_model', 'decoder', 'encoder' ]:
                module_weights = []
                in_projWeightConversion = []
                in_projBiasConversion = []
                for i , (key , perm ) in enumerate(PYTORCH_CKPT_MAPPING[module_name]):
                    if "in_proj" not in key:
                        weight = pytorchWeights['state_dict'][key].numpy()

                        if module_name == "diffusion_model":
                            if "proj_in.weight" in key or "proj_out.weight" in key:
                                #print(i+1," Overriding premuation from constants:\n",key)
                                # This is so the constants.py "diffusion_model" dictionary keeps its legacy state
                                perm = (1,0)

                        if perm is not None:
                            weight = np.transpose(weight , perm )
                        module_weights.append(weight)                          
                    else:
                        if module_name == "text_encoder":
                            # "in_proj" layer of SD2.x is a matrix multiplcation of the query, key, and value layers of SD1.4/5
                            # We will slice this layer into the the three vectors
                            if "weight" in key:
                                # Get the in_proj.weight
                                originalWeight = pytorchWeights['state_dict'][key].float().numpy()

                                queryWeight = originalWeight[:1024, ...]
                                queryWeight = np.transpose(queryWeight, (1,0))

                                keyWeight = originalWeight[1024:2048, ...]
                                keyWeight = np.transpose(keyWeight, (1,0))
                                
                                valueWeight = originalWeight[2048:, ...]
                                valueWeight = np.transpose(valueWeight, (1,0))

                                # Clear local variable to carry forward for bias
                                in_projWeightConversion = []

                                in_projWeightConversion.append(queryWeight) # Query states
                                in_projWeightConversion.append(keyWeight) # Key states
                                in_projWeightConversion.append(valueWeight) # Value states
                            elif "bias" in key:
                                originalBias = pytorchWeights['state_dict'][key].float().numpy()

                                queryBias = originalBias[:1024]

                                keyBias = originalBias[1024:2048]
                                
                                valueBias = originalBias[2048:]

                                # Clear local variable to carry forward for bias
                                in_projBiasConversion = []

                                in_projBiasConversion.append(queryBias) # Query states
                                in_projBiasConversion.append(keyBias) # Key states
                                in_projBiasConversion.append(valueBias) # Value states

                                # add the converted weights/biases in the correct order
                                # Query
                                module_weights.append(in_projWeightConversion[0])
                                module_weights.append(in_projBiasConversion[0])
                                # Key
                                module_weights.append(in_projWeightConversion[1])
                                module_weights.append(in_projBiasConversion[1])
                                # Value
                                module_weights.append(in_projWeightConversion[2])
                                module_weights.append(in_projBiasConversion[2])

                print("Loading weights for ", module_name)
                
                getattr(self, module_name).set_weights(module_weights)
                print("Loaded %d pytorch weights for %s"%(len(module_weights) , module_name))

    def legacyPreparePrompt(self, prompt, batch_size):
        ## Legacy mode

        # First, encode the prompt
        inputs = self.tokenizer.encode(prompt)
        # Then create a tensor of the prompt with a total length 
        if len(inputs) > 77:
            print("Prompt is too long (should be < 77 tokens). Truncating down to 77 tokens")
            inputs = inputs[:77]
        phrase = inputs + [49407] * (77 - len(inputs))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis = 0)

        print("...encoding the tokenized prompt...")
        pos_ids = np.array(list(range(77)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis = 0)

        return phrase, pos_ids
    
    def preparePrompt(self, prompt, batch_size):
        ## Contemporary mode

        # First, encode the prompt
        if isinstance(prompt, str):
            inputs = [prompt]
        # Then tokenize the prompt
        startOfToken = self.tokenizer.encoder["<start_of_text>"]
        endOfToken = self.tokenizer.encoder["<end_of_text>"]
        print("endOfToken:\n",endOfToken)
        allTokens = [[startOfToken] + self.tokenizer.encode(input) + [endOfToken] for input in inputs]
        # Create the empty tensor/numpy array to load the tokens into
        phrase = np.zeros((len(allTokens), MAX_TEXT_LEN), dtype = np.int64)

        for i, tokens in enumerate(allTokens):
            if len(tokens) > MAX_TEXT_LEN:
                tokens = tokens[:MAX_TEXT_LEN]  # Truncate
                tokens[-1] = endOfToken
            phrase[i, :len(tokens)] = np.array(tokens)

        phrase = np.repeat(phrase, batch_size, axis = 0)

        print("...encoding the tokenized prompt...")
        pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)

        return phrase, pos_ids

### Functions

# Create the models we'll use

def get_models(
    img_height = 512,
    img_width = 512,
    download_weights = False,
    preCompiled = None,
    legacy = True
):
    # Memory Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    if legacy is True:
        # Are we using Pre-Stable Diffusion 2.0?
        print("...starting in legacy mode...")
        if preCompiled is None:
            n_h = img_height // 8
            n_w = img_width // 8

            print("Compiling models")

            print("\nLoading metal device\n")

            # Create text encoder model
            input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype = "int32")
            input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype = "int32")
            embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
            text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
            print("Creating text encoder model")

            # Creation diffusion model
            print("Creating diffusion model")
            context = keras.layers.Input((MAX_TEXT_LEN, 768))
            t_emb = keras.layers.Input((320,))
            latent = keras.layers.Input((n_h, n_w, 4))
            unet = LegacyUNetModel()
            diffusion_model = keras.models.Model(
                [latent, t_emb, context], unet([latent, t_emb, context])
            )

            # Create decoder model
            print("Creating decoder model")
            latent = keras.layers.Input((n_h, n_w, 4))
            decoder = Decoder()
            decoder = keras.models.Model(latent, decoder(latent))

            # Create encoder model
            print("Creating encoder model")
            inp_img = keras.layers.Input((img_height, img_width, 3))
            encoder = Encoder()
            encoder = keras.models.Model(inp_img, encoder(inp_img))
            
            # If selected, the program will download pre-made h5 weights

            if download_weights:
                print("\nDownloading weights...")
                text_encoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
                    file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
                )
                diffusion_model_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
                    file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
                )
                decoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
                    file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
                )

                encoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
                    file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
                )

                print("...all weights downloaded!\n\nLoading weights...")

                text_encoder.load_weights(text_encoder_weights_fpath)
                diffusion_model.load_weights(diffusion_model_weights_fpath)
                decoder.load_weights(decoder_weights_fpath)
                encoder.load_weights(encoder_weights_fpath)

                print("...weights loaded!")
            return text_encoder, diffusion_model, decoder , encoder
        else:
            # Load pre-compiled models

            print("Loading pre-compiled models:")

            print("\nLoading metal device\n")

            text_encoder = keras.models.load_model(preCompiled + "/text_encoder")
            print("Text encoder model loaded")
            diffusion_model = keras.models.load_model(preCompiled + "/diffusion_model")
            print("Diffusion model loaded")
            decoder = keras.models.load_model(preCompiled + "/decoder")
            print("Decoder model loaded")
            encoder = keras.models.load_model(preCompiled + "/encoder")
            print("Encoder model loaded")

            return text_encoder, diffusion_model, decoder , encoder
    else:
        # We're using SD 2.0 and newer

        print("...starting in contemporary mode...")

        if preCompiled is None:
            n_h = img_height // 8
            n_w = img_width // 8

            print("Compiling models")

            print("\nLoading metal device\n")

            # Create text encoder model
            input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype = "int32")
            input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype = "int32")
            embeds = OpenCLIPTextTransformer()([input_word_ids, input_pos_ids])
            text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
            print("Creating text encoder model")

            # Creation diffusion model
            print("Creating diffusion model")
            context = keras.layers.Input((MAX_TEXT_LEN, 1024))
            t_emb = keras.layers.Input((320,))
            latent = keras.layers.Input((n_h, n_w, 4))
            unet = UNetModel()
            diffusion_model = keras.models.Model(
                [latent, t_emb, context], unet([latent, t_emb, context])
            )

            # Create decoder model
            print("Creating decoder model")
            latent = keras.layers.Input((n_h, n_w, 4))
            decoder = Decoder()
            decoder = keras.models.Model(latent, decoder(latent))

            #decoderWeights = decoder.weights

            # Create encoder model
            print("Creating encoder model")
            inp_img = keras.layers.Input((img_height, img_width, 3))
            encoder = Encoder()
            encoder = keras.models.Model(inp_img, encoder(inp_img))

            # return created models
            return text_encoder, diffusion_model, decoder , encoder
        else:
            # Load pre-compiled models

            print("Loading pre-compiled models:")

            print("\nLoading metal device\n")

            text_encoder = keras.models.load_model(preCompiled + "/text_encoder")
            print("Text encoder model loaded")
            diffusion_model = keras.models.load_model(preCompiled + "/diffusion_model")
            print("Diffusion model loaded")
            decoder = keras.models.load_model(preCompiled + "/decoder")
            print("Decoder model loaded")
            encoder = keras.models.load_model(preCompiled + "/encoder")
            print("Encoder model loaded")

            return text_encoder, diffusion_model, decoder , encoder
