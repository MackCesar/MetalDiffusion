### Main Modules
import sys

### Console GUI
from rich import print
from tqdm.auto import tqdm

### Traceback
try:
    import traceback2 as traceback
except ImportError:
    print("Warning: Import error for Pretty Traceback")
    pass    # no need to fail because of missing dev dependency

### Math Modules
import numpy as np
import random

### Machine Learning Modules
import torch
import torchvision.transforms as transforms
# Standard Pipeline
from diffusers import DiffusionPipeline
#from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
#from diffusers import StableDiffusionPipeline
from stableDiffusionDiffusers.communityPipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from stableDiffusionDiffusers.communityPipelines.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
# ControlNet
# from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import ControlNetModel
from stableDiffusionDiffusers.communityPipelines.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from stableDiffusionDiffusers.communityPipelines.pipeline_controlnet import StableDiffusionControlNetPipeline
# Community Pipelines
from stableDiffusionDiffusers.communityPipelines.stable_diffusion_controlnet_reference import StableDiffusionControlNetReferencePipeline
# Components
from diffusers import AutoencoderKL
from transformers import CLIPTextModel

# Schedulers
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

### Weights Loading
from safetensors import safe_open

### Prompt Engineering
from compel import Compel
from compel import DiffusersTextualInversionManager

### Memory Management
import gc #Garbag Collector

### Token Merging
import tomesd

### Modules for image building
from PIL import Image
import cv2 #OpenCV

class StableDiffusionDiffusers:
    ### Base class/object for Stable Diffusion
    def __init__(
        self,
        imageHeight = 512,
        imageWidth = 512,
        jit_compile = False,
        weights = None,
        legacy = True,
        VAE = "Original",
        textEmbeddings = None,
        mixedPrecision = False,
        optimizer = None,
        device = "mps",
        controlNet = [False, None], # [0] = Use ControlNet? [1] = ControlNet Weights [2] = Input [3] = Strength
        LoRAs = None,
        tokenMergingStrength = 0.5,
        CLIPSkip = 0
    ):

        ### Step 1: Establish image dimensions for UNet ###
        ## requires multiples of 2**7, 2 to the power of 7
        self.imageHeight = round(imageHeight / 128) * 128
        self.imageWidth = round(imageWidth / 128) * 128

        ### Step 2: Text Embeddings
        if textEmbeddings == None:
            print("\nIgnorning Text Embeddings")
            self.textEmbeddings = None
        else:
            print("\nUsing Text Embeddings")
            # Create unique tokens based on filename
            self.textEmbeddings, self.textEmbeddingsTokens = loadTextEmbeddings(textEmbeddings)
        
        ### Step 3: LoRAs
        if len(LoRAs) <= 1:
            print("\nIgnoring LoRAs")
            self.LoRAs = []
            self.LoRAPath = None
        else:
            print("\nUsing LoRAs")
            self.LoRAs = LoRAs
            self.LoRAPath = None
        
        ### Step 4: Create Models

        ## Step 4.1: ControlNet Pipeline
        if controlNet[0] == True:
            print("\nUsing ControlNet", controlNet[1])
            self.controlNetWeights = controlNet[1]
            self.controlNet = True
        else:
            self.controlNetWeights = None
            self.controlNet = None

        ## Step 4.2: Store Weights
        self.weights = weights

        ## Step 4.3: VAE
        if VAE != "Original":
            self.VAE = AutoencoderKL.from_single_file(VAE)
        else:
            self.VAE = None
        
        ## Step 4.4: Token Merging
        self.tokenMergingStrength = tokenMergingStrength
        if tokenMergingStrength > 0:
            print("\nUsing Token Merging [bold](ToMe)[/bold] at ",str(tokenMergingStrength * 100),"percent strength")

        ### Step 4.3: Load Pipeline(s)
        self.pipeline = None
        self.imageToImagePipeline = False
        self.torchDevice = device

        #### Precision
        self.mixedPrecision = mixedPrecision
        self.dtype = torch.float32
        self.CLIPSkip = CLIPSkip

        self.compileModels()

        ### Step 4: Graph Models ###
        self.jitCompile = jit_compile
        if self.jitCompile is True:
            print("\nCompling model...")
            self.pipeline.unet = torch.compile(
                self.pipeline.unet,
                mode = "reduce-overhead"
            )
            print("...[green]done![/green]")

        ## Cache
        self.promptCache = None
        self.negativePromptCache = None
        self.encodedPrompt = None
        self.encodedNegativePrompt = None
        self.batch_size = None
        self.controlNetCache = None

    """
    Compile Pipelines (models)
    """
    def compileModels(
            self
            ):
        
        self.changePolicy(self.mixedPrecision)

        ## Memory Efficiency
        self.pipeline = None
        clearMemory()

        # Load ControlNet Pipeline if true
        if self.controlNet is True:
            ## Memory Efficiency
            self.controlNet = None
            clearMemory()

            ## Create pipeline
            if "safetensors"in self.controlNetWeights or "ckpt" in self.controlNetWeights:
                print("Loading",self.controlNetWeights,"via single file safetensors...")
                self.controlNet = ControlNetModel.from_pretrained(
                    self.controlNetWeights
                    )
                print("...[green]done![/green]")
            else:
                if "Reference Only" in self.controlNetWeights:
                    print("Using [bold]Reference Only[/bold] ContorlNet...")
                    self.controlNetWeights = "Reference Only"
                else:
                    print("Loading",self.controlNetWeights,"via pretrained diffusers...")
                    self.controlNet = ControlNetModel.from_pretrained(
                        self.controlNetWeights
                        )
                print("...[green]done![/green]")
            self.controlNet.to(self.torchDevice)
        
        # Load Main Pipeline
        if "safetensors" in self.weights or "ckpt" in self.weights:
            print("\nLoading", self.weights,"weights via single file safetensors...")
            if self.controlNet is None:
                if self.imageToImagePipeline is True:
                    print("...loading Image To Image pipeline...")
                    self.pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                        self.weights
                        )
                else:
                    print("...loading standard pipeline...")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        self.weights
                        )
            else:
                if self.imageToImagePipeline is True:
                    print("...loading ControlNet Image To Image pipeline...")
                    self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        self.weights,
                        controlnet = self.controlNet,
                        torch_dtype = self.dtype
                        )
                else:
                    print("...loading ControlNet pipeline...")
                    if self.controlNetWeights == "Reference Only":
                        self.pipeline = StableDiffusionControlNetReferencePipeline.from_pretrained(
                            self.weights,
                            controlnet = self.controlNet,
                            torch_dtype = self.dtype
                        )
                    else:
                        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                            self.weights,
                            controlnet = self.controlNet,
                            torch_dtype = self.dtype
                            )
            print("...[green]done![/green]")
        else:
            print("\nLoading",self.weights,"weights via pretrained diffusers...")
            if self.controlNet is None:
                if self.imageToImagePipeline is True:
                    print("...loading Image To Image pipeline...")
                    self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        self.weights,
                        torch_dtype = self.dtype,
                        safety_checker = None
                        )
                else:
                    print("...loading standard pipeline...")
                    ### CLIP Skip
                    if self.CLIPSkip > 0:
                        print("...using CLIP Skip:",self.CLIPSkip,"...")
                    textEncoder = CLIPTextModel.from_pretrained(
                        self.weights,
                        subfolder = "text_encoder",
                        num_hidden_layers = (12 - self.CLIPSkip),
                        use_safetensors = True,
                        torch_dtype = self.dtype
                    )

                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.weights,
                        text_encoder = textEncoder,
                        torch_dtype = self.dtype,
                        use_safetensors = True,
                        safety_checker = None
                    )

                    if self.VAE != None:
                        print("...using",self.VAE,"as VAE...")
                        self.pipeline.VAE = self.VAE
            else:
                if self.imageToImagePipeline is True:
                    print("...loading ControlNet Image To Image pipeline...")
                    self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        self.weights,
                        controlnet = self.controlNet,
                        torch_dtype = self.dtype,
                        safety_checker = None
                        )
                else:
                    print("...loading ControlNet pipeline...")
                    if self.controlNetWeights == "Reference Only":
                        self.pipeline = StableDiffusionControlNetReferencePipeline.from_pretrained(
                            self.weights,
                            controlnet = self.controlNet,
                            torch_dtype = self.dtype,
                            safety_checker = None
                            )
                    else:
                        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                            self.weights,
                            controlnet = self.controlNet,
                            torch_dtype = self.dtype,
                            safety_checker = None
                            )
            print("...[green]done![/green]")
        
        # LoRAs
        if len(self.LoRAs) > 0:
            print("\nLoading [bold]LoRAs[/bold]...")
            self.pipeline.to(self.torchDevice)
            if self.LoRAPath == None:
                self.LoRAPath = self.LoRAs[0]
                del self.LoRAs[0]
            for lora in self.LoRAs:
                print("...loading",lora)
                self.pipeline.load_lora_weights("./"+self.LoRAPath, weight_name = lora)
                print("...loaded!")

        # Optimizations
        ## Move to GPU - Old
        #self.pipeline.to(self.torchDevice)
        ## Model CPU Offload, saves memory
        ## See: https://huggingface.co/docs/diffusers/optimization/fp16#model-offloading-for-fast-inference-and-memory-savings
        self.pipeline.enable_model_cpu_offload(gpu_id = self.torchDevice)

        ## Attention, the most memory hungry version
        #self.pipeline.enable_attention_slicing(1)
        #self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        ## Note: Seems like we don't need to do this. See: https://pytorch.org/blog/accelerated-diffusers-pt-20/

        ## Safety Checker
        self.pipeline.safety_checker = None
        
        # Text Embeddings
        if self.textEmbeddings != None:
            print("\nLoading [bold]text embeddings[/bold]...")
            for index, embedding in enumerate(self.textEmbeddings):
                print("...loading",self.textEmbeddingsTokens[index])
                path = "./" + embedding
                print("from:", path)
                self.pipeline.load_textual_inversion(
                    path,
                    token = self.textEmbeddingsTokens[index]
                    )
                print(self.textEmbeddingsTokens[index],"loaded!")
            print("...done! All text embeddings loaded.")
        
        # Token Merging
        if self.tokenMergingStrength > 0:
            print("Applying token merging...")
            tomesd.apply_patch(self.pipeline, ratio = 0.5)
            print("...done!")
        
        ## Final Messages
        if self.torchDevice == "mps":
            print("\nDiffusers on GPU via :mechanical_arm: [white]Metal[/white] :mechanical_arm: ready!")
        else:
            print("\nDiffusers on CPU ready!")
    """
    Change Policy
    """
    def changePolicy(self, policy):
        if policy is True:
            if self.dtype != torch.float16:
                print("\n...changing to mixed precision (FP16)...")
                self.dtype = torch.float16
                self.compileModels()
            else:
                print("\n...using mixed precision (FP16)...")
        else:
            if self.dtype != torch.float32:
                print("\n...changing to regular precision (FP32)...")
                self.dtype = torch.float32
                torch.set_float32_matmul_precision('high')
                self.compileModels()
            else:
                print("\n...using regular precision (FP32)...")
                self.dtype = torch.float32
                torch.set_float32_matmul_precision('high')
    """
    Generate and image, the key function
    """
    def generate(
        self,
        prompt,
        negativePrompt = None,
        batch_size = 1,
        num_steps = 25,
        unconditional_guidance_scale = 7.5,
        temperature = 1,
        seed = None,
        input_image = None, # expecting file path as a string or np.ndarray
        input_image_strength = 0.5,
        input_mask = None, # expecting a file path as a string
        sampler = None,
        controlNetStrength = 1,
        controlNetImage = None,
        controlNetCache = False,
        vPrediction = False,
        LoRAStrength = 0.5
    ):
        ## Memory Efficiency
        print("\n...cleaning up memory...")
        clearMemory()
        print("...getting to work...")

        ### Step 1: Prepare Seed
        if seed is None:
            print("...generating random seed...")
            seed = random.randint(1000, sys.maxsize)
            seed = int(seed)
            seed = torch.Generator(device = self.torchDevice).manual_seed(seed)
        else:
            print("...preparing seed...")
            seed = torch.Generator(device = self.torchDevice).manual_seed(int(seed))
        
        ### Step 2: Tokenize prompts
        ## Already done by diffusers
        ## Compel is a python module that parses prompt weighting

        textualInversionManager = DiffusersTextualInversionManager(self.pipeline)

        compel = Compel(
            tokenizer = self.pipeline.tokenizer,
            text_encoder = self.pipeline.text_encoder,
            textual_inversion_manager = textualInversionManager
        )

        print("...preparing prompt...")
        promptEmbedding = compel.build_conditioning_tensor(prompt)
        print("...preparing negative prompt...")
        negativeEmbedding = compel.build_conditioning_tensor(negativePrompt)
        
        ### Step 3: Prepare the input image, if it was given
        if input_image is not None:
            print("...preparing input image...")
            print("...input image type:",type(input_image),"...")
            if self.imageToImagePipeline is False:
                print("...updating pipeline...")
                self.imageToImagePipeline = True
                self.compileModels()
                print("\n...pipeline updated...")
            if isinstance(input_image, np.ndarray):
                ### Because of the way our image manipulation for video works
                ### the image has been rearranged into cv2, which is BGR
                ### This is assuming, of course, that we've given a numpy array from utilities/videoUtilities.py
                print("...converting image back to PIL...")
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = Image.fromarray(input_image)
            if input_image.height != self.imageHeight:
                print("...resizing input image to render size...")
                input_image = input_image.resize((self.imageWidth,self.imageHeight), resample = Image.BICUBIC)
            if input_image.width != self.imageWidth:
                print("...resizing input image to render size...")
                input_image = input_image.resize((self.imageWidth,self.imageHeight), resample = Image.BICUBIC)
            print("...using input image size:", input_image.size)
            #print(type(input_image))
            if batch_size > 1:
                input_image = [input_image] * batch_size
        else:
            if self.imageToImagePipeline is True:
                print("...updating pipeline...")
                self.imageToImagePipeline = False
                self.compileModels()
                print("\n...pipeline updated...")
        
        ### Step 4: Create time steps
        ## Already done by diffusers
        if len(self.LoRAs) > 0:
            useKarrasSigmas = True
        else:
            useKarrasSigmas = False

        ### Step 5: Load Sampler
        ## List of samplers: sampleChoices = ["Basic", "DDIM", "DDPM", "DPM Solver", "Euler", "Euler A", "LMS", "PNDM"]
        if sampler == "DDIM":
            print("...loading [blue]DDIM[/blue] sampler...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "DDPM":
            print("...loading [blue]DDPM[/blue] sampler...")
            self.pipeline.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "DPM Solver":
            print("...loading [blue]DPM Solver[/blue] sampler...")
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config, use_karras_sigmas = useKarrasSigmas)
        elif sampler == "Euler":
            print("...loading [blue]Euler[/blue] sampler...")
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "Euler A":
            print("...loading [blue]Euler Ancestral[/blue] sampler...")
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "LMS":
            print("...loading [blue]LMS[/blue] sampler...")
            self.pipeline.scheduler = LMSDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "PNDM":
            print("...loading [blue]PNDM[/blue] sampler...")
            self.pipeline.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
        elif sampler == "UniPC":
            print("...loading [blue]UniPC Multistep[/blue] sampler...")
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        else:
            print("...using [blue]basic[/blue] sampler...")
            self.pipeline.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
        
        ### Step 6: ControlNet
        if self.controlNet != None:
            if isinstance(controlNetStrength, list):
                controlNetStrength = controlNetStrength[0]
                controlNetGuessMode = True
                print("...using [orange]guess mode[/orange] for ControlNet...")
                print("...using ControlNet [bold]strength[/bold] of",controlNetStrength,"...")
            else:
                print("...using ControlNet [bold]strength[/bold] of",controlNetStrength,"...")
                controlNetGuessMode = False
        
        ### Step 7: LoRAs
        if len(self.LoRAs) > 0:
            loraScale = {"scale": LoRAStrength}
            print("...using LoRA [bold]strength[/bold] of",loraScale)
        else:
            loraScale = None
        
        ### Step 8: Final Memory Clearing
        clearMemory()

        ### Step 9: Start Diffusion
        print("Starting diffusion:")
        if self.imageToImagePipeline is True:
            if self.controlNet is None:
                decoded = self.pipeline(
                    prompt_embeds = promptEmbedding,
                    negative_prompt_embeds = negativeEmbedding,
                    image = input_image,
                    strength = input_image_strength,
                    num_inference_steps = num_steps,
                    guidance_scale = unconditional_guidance_scale,
                    generator = seed,
                    num_images_per_prompt = batch_size,
                    cross_attention_kwargs = loraScale
                )
            else:
                decoded = self.pipeline(
                    prompt_embeds = promptEmbedding,
                    negative_prompt_embeds = negativeEmbedding,
                    image = input_image,
                    control_image = controlNetImage,
                    strength = input_image_strength,
                    num_inference_steps = num_steps,
                    guidance_scale = unconditional_guidance_scale,
                    generator = seed,
                    num_images_per_prompt = batch_size,
                    controlnet_conditioning_scale = controlNetStrength,
                    guess_mode = controlNetGuessMode,
                    cross_attention_kwargs = loraScale
                )
        else:
            if self.controlNet is None:
                decoded = self.pipeline(
                    prompt_embeds = promptEmbedding,
                    negative_prompt_embeds = negativeEmbedding,
                    height = self.imageHeight,
                    width = self.imageWidth,
                    num_inference_steps = num_steps,
                    guidance_scale = unconditional_guidance_scale,
                    generator = seed,
                    num_images_per_prompt = batch_size,
                    cross_attention_kwargs = loraScale
                )
            else:
                decoded = self.pipeline(
                    prompt_embeds = promptEmbedding,
                    negative_prompt_embeds = negativeEmbedding,
                    image = controlNetImage,
                    num_inference_steps = num_steps,
                    guidance_scale = unconditional_guidance_scale,
                    generator = seed,
                    num_images_per_prompt = batch_size,
                    controlnet_conditioning_scale = controlNetStrength,
                    guess_mode = controlNetGuessMode,
                    cross_attention_kwargs = loraScale
                )
        
        ### Step 10: Decoding stage

        # Done by Diffusers
        
        ### Step 11: Merge inpainting
        
        # Done by Diffusers

        ### Step 12: Memory cleanup
        clearMemory()

        ### Step 13: Return final image(s) as a list
        print("\n...decoding latent image...")
        images = []
        for image in decoded.images:
            images.append(image)

        return images
    
    def setWeights(
            self,
            modelLocation,
            VAELocation
    ):
        self.weights = modelLocation
        self.compileModels()

### Safety Checker
def dummy(images, **kwargs):
    #print("Number of images: ",len(images))
    safetyResult = [False] * len(images)
    return images, safetyResult

def loadTextEmbeddings(
        allEmbeddings = None
):
    """
    This function determines the unique token from the file name of the text embedding, to match how the TensorFlow implementation works
    """
    if allEmbeddings == None:
        print("[bold red]FATAL ERROR[/bold red]\nNo Text Embeddings passed, so none to load")
        return
    
    # save file path into seperate location
    embeddingsPath = allEmbeddings[0]
    # delete file path from list
    del allEmbeddings[0]
    
    finalTextEmbeddings = []
    finalTokens = []
    
    for textEmbedding in allEmbeddings:
        if "pt" in textEmbedding:
            textEmbeddingName = textEmbedding.replace(".pt","")
        elif "bin" in textEmbedding:
            textEmbeddingName = textEmbedding.replace(".bin","")

        # Make the token lowercase
        token = "<" + textEmbeddingName.lower() + ">"
        #print("Unique Token: ", token)
        #print("In:",embeddingsPath + textEmbedding)

        finalTextEmbeddings.append(embeddingsPath + textEmbedding)
        finalTokens.append(token)
    
    return finalTextEmbeddings, finalTokens

def clearMemory():
    gc.collect()
    try:
        torch.mps.empty_cache()
        print("...PyTorch cache cleared...")
    except Exception as e:
        print("...PyTorch cache is empty...")
    gc.collect()

if __name__ == "__main__":
    print("\n\n[bold]Hello![/bold] Welcome to the Diffusers and PyTorch version of MetalDiffusion.")

    print("\nThis time, we're trying out SDXL")
    
    print("Running diffusers now!")

    #prompt = "a gorgeous photograph of an astronaut riding a horse, science fiction, highly detailed, in the style of Jakub Rozalski"
    prompt = "beautiful render of planets, some stars in the sky, stunning planets, brilliant star, galaxy in the distance, multiple planets, asteroid belt, comets, highly detailed, science fiction, digital painting, masterpiece, cinematic, gorgeous, world, blue planets, colorful"
    negativePrompt = "landscape, mountains, horizon, ground, surface"
    inputImage = Image.open("diffPyTorch/finalWideScreen.png")
    seed = random.randint(0, 2 ** 31)
    print("\nCreating generator")

    """generator = StableDiffusionDiffusers(
        imageWidth = 1024,
        imageHeight = 512,
        weights = "./models/diffusers/DreamShaper",
        jit_compile = False
    )"""

    print("Creating base model...")
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors = True
    )
    base.to("mps")

    print("Creating refiner model...")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2 = base.text_encoder_2,
        vae = base.vae,
        use_safetensors=True
    )
    refiner.to("mps")

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    print("\nGenerating image")

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    """images = generator.generate(
        prompt = prompt,
        negativePrompt = negativePrompt,
        batch_size = 1,
        num_steps = 25,
        seed = seed,
        unconditional_guidance_scale = 2.1,
        input_image = None,
        input_image_strength = 0.2,
        sampler = "Euler"
    )"""

    print("\nDone! Saving image(s)")

    location = "stableDiffusionDiffusers/final0000"+str(seed)+".png"
    image.save(location)

    """i = 0
    for image in images:
        location = "stableDiffusionDiffusers/final0000"+str(seed)+".png"
        image.save(location)
        i += 1"""