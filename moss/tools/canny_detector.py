try:
    from controlnet_aux.processor import Processor
except:
    raise ImportError("An additional package `controlnet_aux` is required when using PoseDetector Tool")

from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field

import numpy as np


class CannyDetectorInput(BaseModel):
    image : str = Field(title="image", description="The url of the image containing the object to be detected for canny.")


class CannyDetector(BaseTool):
    name: str = "canny_detector"
    description: str = "Extracts the canny in an image, returns a canny image"
    args_schema: Type[BaseModel] = CannyDetectorInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        image_obj = np.array(Image.open(image))
        return Image.fromarray(Processor("canny")(image_obj))

    async def _arun(self, image: str) -> Image.Image:
        """_arun is not implemented yet. Call sync run instead."""
        return self._run(image)