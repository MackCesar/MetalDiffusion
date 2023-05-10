# TensorFlow Modules
import tensorflow as tf
from tensorflow import keras

class BasicSampler():
    def __init__(
        self,
        model = None,
        timesteps = tf.range(1, 1000, 1000 // 50),
        batchSize = 1,
        seed = 1990,
        inputImage = None, # Expecting a tensor
        inputMask = None, # Expecting a tensor
        inputImageStrength = 0.5,
        temperature = 1,
        AlphasCumprod = None
    ):
        print("...starting Basic Sampler...")
        self.model = model
        self.timesteps = timesteps
        self.batchSize = batchSize
        self.seed = seed
        self.inputImage = inputImage
        self.inputMask = inputMask
        self.inputImageStrength = inputImageStrength
        self.inputImageNoise_T = self.timesteps[ int(len(self.timesteps) * self.inputImageStrength) ]
        self.temperature = temperature
        self.AlphasCumprod = AlphasCumprod

        self.latent, self.alphas, self.alphas_prev = self.getStartingParameters(
            self.timesteps,
            self.batchSize,
            seed,
            inputImage = self.inputImage,
            inputImageNoise_T = self.inputImageNoise_T
        )
        
        if self.inputImage is not None:
            self.timesteps = self.timesteps[: int( len(self.timesteps) * self.inputImageStrength ) ]
        
        print("...sampler ready...")
    
    def addNoise(
        self,
        x,
        t,
        noise = None,
        DType = tf.float32
    ):
        batch_size , w , h = x.shape[0] , x.shape[1] , x.shape[2]
        if noise is None:
            noise = tf.random.normal((batch_size,w,h,4), dtype = DType)
        sqrt_alpha_prod = self.AlphasCumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.AlphasCumprod[t]) ** 0.5

        return  sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
            
    def getStartingParameters(
        self,
        timesteps,
        batchSize,
        seed,
        inputImage = None,
        inputImageNoise_T = None
    ):
        # Use floor division to get minimum height/width of image size
        # for the Diffusion and Decoder models
        floorDividedImageHeight = self.model.imageHeight // 8
        floorDividedImageWidth = self.model.imageWidth // 8

        alphas = [self.AlphasCumprod[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        if inputImage is None:
            # Create a random input image from noise
            latent = tf.random.normal(
                (batchSize, floorDividedImageHeight, floorDividedImageWidth, 4),
                seed = seed
            )
        else:
            # Encode the given image
            latent = self.model.encoder(inputImage, training = False)
            # Repeat it within the tensor for the given batch size
            latent = tf.repeat(latent , batchSize , axis = 0)
            # Noise the image
            latent = self.addNoise(latent, inputImageNoise_T)
        
        return latent, alphas, alphas_prev
    
    def get_x_prev_and_pred_x0(
            self,
            x,
            e_t,
            index,
            a_t,
            a_prev,
            temperature,
            seed
    ):
        sigma_t = tf.constant(0.0)
        sqrt_one_minus_at = tf.sqrt(tf.constant(1.0) - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / tf.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = tf.sqrt(tf.constant(1.0) - a_prev - tf.square(sigma_t)) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed = seed) * temperature
        x_prev = tf.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0
    
    # Keras Version
    def sample(
            self,
            context,
            unconditionalContext,
            unconditionalGuidanceScale
    ):
        tf.print("...sampling:")

        # Progress Bar set-up
        progbar = tf.keras.utils.Progbar(len(self.timesteps))
        iteration = 0

        # Iteration loop
        for index, timestep in list(enumerate(self.timesteps))[::-1]:

            latentPrevious = self.latent

            # Establish timestep embedding
            t_emb = self.timestepEmbedding(float(timestep))
            t_emb = tf.repeat(t_emb, self.batchSize, axis = 0)

            # Get unconditional (negative prompt) latent image
            unconditionalLatent = self.model.diffusion_model(
                [self.latent, t_emb, unconditionalContext],
                training = False
            )
            # Get conditional (positive prompt) latent image
            self.latent = self.model.diffusion_model(
                [self.latent, t_emb, context],
                training = False
            )

            # Combine the two latent images, the et
            self.latent = unconditionalLatent + unconditionalGuidanceScale * (self.latent - unconditionalLatent)

            # Alphas, the sigma
            a_t, a_prev = self.alphas[index], self.alphas_prev[index]

            """# Predictions
            predictV = (latentPrevious - tf.sqrt(tf.constant(1.0) - a_t) * self.latent) / tf.sqrt(
                a_t
            )
            self.latent = (
                self.latent * tf.sqrt(1.0 - a_prev) + tf.sqrt(a_prev) * predictV
            )"""

            # Predictions
            predictV = (latentPrevious - tf.sqrt(tf.constant(1.0) - a_t) * self.latent) / tf.sqrt(
                a_t
            )
            self.latent = (
                self.latent * tf.sqrt(1.0 - a_prev) + tf.sqrt(a_prev) * predictV
            )

            # Keras Progress Bar Update
            iteration += 1
            progbar.update(iteration)

        tf.print("...finished! Returning latent image...")

        return self.latent
    
    
    def getModelOutput(
        self,
        latent,
        inputTimesteps,
        context,
        unconditionalContext,
        unconditionalGuidanceScale,
        batch_size,
    ):

        # Establish timestep embedding
        t_emb = self.timestepEmbedding(float(inputTimesteps))
        t_emb = tf.repeat(t_emb, batch_size, axis = 0)

        # Get unconditional (negative prompt) latent image
        unconditionalLatent = self.model.diffusion_model(
            [latent, t_emb, unconditionalContext],
            training = False
        )
        # Get conditional (positive prompt) latent image
        latent = self.model.diffusion_model(
            [latent, t_emb, context],
            training = False
        )

        # Combine the images and return the result
        return unconditionalLatent + unconditionalGuidanceScale * (
            latent - unconditionalLatent
        )
    
    def timestepEmbedding(
        self,
        timesteps,
        dimensions = 320,
        max_period = 10000.0
    ):
        half = dimensions // 2
        freqs = tf.math.exp(
            -tf.math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timesteps], dtype = tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding