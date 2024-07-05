# from diffusers import DiffusionPipeline
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


