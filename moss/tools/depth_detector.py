try:
    from controlnet_aux.processor import Processor
except:
    raise ImportError("An additional package `controlnet_aux` is required when using DepthDetector Tool")

from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field


class DepthDetectorInput(BaseModel):
    image : str = Field(title="image", description="The url of the image containing the object to be detected for depth.")


class DepthDetector(BaseTool):
    name: str = "depth_detector"
    description: str = "Extracts the depth in an image, returns a depth image"
    args_schema: Type[BaseModel] = DepthDetectorInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        return Processor("depth_zoe")(Image.open(image))

    async def _arun(self, image: str) -> Image.Image:
        """_arun is not implemented yet. Call sync run instead."""
        return self._run(image)