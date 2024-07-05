# About Diffusers

For the math of transformer models, materials can be found easily in many places. Here, the article dysmystified some important topics in transformer models, https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/.

Some articles were listed here for reference, their content was primaryly about the arithmetic issues related to transformer models. 

As for the arithmetic intensiry required to carry out transformer inference, some details have been presented in an article, https://kipp.ly/transformer-inference-arithmetic/. Besides the discussion on FLOPs, extra analysis was made on the memory costs as well. In addition, reference can be found in another artile, https://www.adamcasson.com/posts/transformer-flops.

In analysis, the roof-line model has always been used, and in https://www.baseten.co/blog/llm-transformer-inference-guide/, there is some analysis work on LLM inference and performance. Furthermore, benchmarking tests were conducted to figure out the inference performance of LLM models, as shown in https://www.baseten.co/blog/benchmarking-fast-mistral-7b-inference/. 


It might be expected to train the stable diffusion models, and descriptions can be found in https://github.com/explainingai-code/StableDiffusion-PyTorch.

The code snippet below shows how to load the SD.v1-5 model (from https://blog.csdn.net/qq_38423499/article/details/137158458).
'''
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
'''

'''
from diffusers import DiffusionPipeline
import torch
from PIL.Image import Image


prompt = "An astronaut riding a green horse"

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

flagCompile=True

seed = torch.Generator("cuda").manual_seed(42)

# Return the status of the base inferencing phase
print("The case without pytorch compile is used for control ")
for i in range(5):
        image = pipe(prompt=prompt, generator=seed, output_type="latent").images

print("The case with pytorch compile for its performance in improving base model")
# Run torch compile
if flagCompile==True:
        pipe.unet.to(memory_format=torch.channels_last)
        #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
for i in range(5):
        image1 = pipe(prompt=prompt, generator=seed, output_type="latent").images
# image.save("SDXL-Base.png")


# Return the status of the refine inferencing phase
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

print("The case without pytorch compile is used for control ")
for i in range(5):
        image = pipe(prompt=prompt, generator=seed, image=image1).images

print("The case with PyTorch compile for its performacne in improving refine model ")
# Run torch compile
if flagCompile==True:
        pipe.unet.to(memory_format=torch.channels_last)
        #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)

for i in range(5):
        image2 = pipe(prompt=prompt, generator=seed, image=image1).images
# image.save("SDXL-Base-Refiner.png")

'''
