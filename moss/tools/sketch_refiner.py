from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field
from ..configs import SD_TOKEN

import io
import requests


class SketchRefinerInput(BaseModel):
    sketch  : str = Field(
        title="image",
        description="The url of the sketch image to refine, could be a pose image, lineart or something like these"
    )
    prompt : str = Field(title="prompt", description="The prompt of generating image according to the sketch")


class SketchRefiner(BaseTool):
    name: str = "sketch_refiner"
    description: str = "Generate a nice image according to the prompt based on a sketch image"
    args_schema: Type[BaseModel] = SketchRefinerInput

    def _run(self, sketch: str, prompt: str) -> Image.Image:
        """Use the tool."""
        headers = {
            "authorization": f"Bearer {SD_TOKEN}",
            "accept": "image/*"
        }
        files = { "image": open(sketch, "rb") }
        data = {
            "prompt": prompt,
            "control_strength": 0.7,
            "output_format": "png"
        }
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/control/sketch",
            headers=headers,
            files=files,
            data=data
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise Exception(str(response.json()))



    async def _arun(self, image: str) -> Image.Image:
        """_arun is not implemented yet. Call sync run instead."""
        return self._run(image)