from typing import Type
from huggingface_hub import AsyncInferenceClient, InferenceClient # type: ignore
from langchain.tools import BaseTool
from PIL import Image, ImageDraw
from pydantic.v1 import BaseModel, Field
from ..configs import HF_TOKEN

import random


class ImageSegmenterInput(BaseModel):
    image: str = Field(description="The url of the image to segment")


class ImageSegmenter(BaseTool):
    name: str = "image_segmenter"
    description: str = "Perform an image segmentation"
    args_schema: Type[BaseModel] = ImageSegmenterInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        client = InferenceClient(token=HF_TOKEN)
        image_obj = Image.open(image)
        predicted = client.image_segmentation(image)
        for p in predicted:
            mask = p.pop("mask")
            layer = Image.new(
                "RGBA", mask.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 155)
            )
            image_obj.paste(layer, (0, 0), mask)
        return image_obj


    async def _arun(self, image: str) -> Image.Image:
        """Use the tool."""
        client = AsyncInferenceClient(token=HF_TOKEN)
        image_obj = Image.open(image)
        predicted = await client.image_segmentation(image)
        for p in predicted:
            mask = p.pop("mask")
            layer = Image.new(
                "RGBA", mask.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 155)
            )
            image_obj.paste(layer, (0, 0), mask)
        return image_obj
