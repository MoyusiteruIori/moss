try:
    from controlnet_aux.processor import Processor
except:
    raise ImportError("An additional package `controlnet_aux` is required when using PoseDetector Tool")

from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field


class PoseDetectorInput(BaseModel):
    image : str = Field(title="image", description="The url of the image containing the object to be detected for pose.")


class PoseDetector(BaseTool):
    name: str = "pose_detector"
    description: str = "Extracts the pose of the object in an image, returns a pose image"
    args_schema: Type[BaseModel] = PoseDetectorInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        return Processor("openpose_full")(Image.open(image))

    async def _arun(self, image: str) -> Image.Image:
        """_arun is not implemented yet. Call sync run instead."""
        return self._run(image)