from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux.processor import Processor
import numpy as np
import torch
import cv2
from PIL import Image

# # download an image
# image = load_image(
#     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
# )
# image = np.array(image)

# # get canny image
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)
# canny_image.save(f"canny.png")

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# pipe.enable_model_cpu_offload()

# # generate image
# generator = torch.manual_seed(0)
# image = pipe(
#     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
# ).images[0]
# if isinstance(image, Image.Image):
#     image.save(f"test.png")

#########################################################################################################################

# download an image
image = Image.open("pose.png")
image = np.array(image)

# get pose image
pose_image = Processor("openpose_full")(image)
pose_image.save("pose.png")

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# pipe.enable_model_cpu_offload()

# # generate image
# generator = torch.manual_seed(0)
# image = pipe(
#     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=pose_image
# ).images[0]
# if isinstance(image, Image.Image):
#     image.save(f"test.png")


#########################################################################################################################

# # download an image
# image = Image.open("man.png")
# image = np.array(image)

# # get pose image
# dep_image = Image.fromarray(Processor("depth_zoe")(image))
# dep_image.save("dep.png")

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# pipe.enable_model_cpu_offload()

# # generate image
# generator = torch.manual_seed(0)
# image = pipe(
#     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=dep_image
# ).images[0]
# if isinstance(image, Image.Image):
#     image.save(f"test.png")