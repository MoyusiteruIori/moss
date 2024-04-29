from typing import Dict, Literal, Tuple
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field

import torch


class ImageToImageGeneratorInput(BaseModel):
    prompt : str = Field(
        title="prompt",
        description="The prompt of generating an image"
    )
    source_image : str = Field(
        title="source_image",
        description="URL of the original image used to guide image generation."
    )
    control_type : Literal["pose", "canny", "depth"] = Field(
        title="control_type",
        description="This field is used to guide how the source image influences the generation of" 
        "the target image. \"pose\" indicates generating a new image based on the pose of the object in the original "
        "image. \"canny\" indicates generating a new image based on the contours in the original image. "
        "\"depth\" indicates generating a new image based on the depth relationships in the original image."
    )


class ImageToImageGenerator(BaseTool):
    name: str = "image_to_image_generator"
    description: str = "Generate an image according to a source image, a piece of prompt, and a control type."
    args_schema: Type[BaseModel] = ImageToImageGeneratorInput

    def _run(self, prompt: str, source_image: str, control_type: Literal["pose", "canny", "depth"]) -> Image.Image:
        """Use the tool."""
        model_map: Dict[str, Tuple[str, str]] = {
            "pose":  "lllyasviel/sd-controlnet-openpose",
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth"
        }
        controlnet_name = model_map[control_type]
        controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        image = pipe(
            prompt, num_inference_steps=20, generator=torch.manual_seed(0), image=Image.open(source_image)
        ).images[0]
        if isinstance(image, Image.Image):
            return image
        else:
            raise Exception("Generation failed")

    async def _arun(self, image: str) -> Image.Image:
        """_arun is not implemented yet. Call sync run instead."""
        return self._run(image)