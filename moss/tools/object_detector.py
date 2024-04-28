from typing import Type
from huggingface_hub import AsyncInferenceClient, InferenceClient # type: ignore
from langchain.tools import BaseTool
from PIL import Image, ImageDraw
from pydantic.v1 import BaseModel, Field
from ..configs import HF_TOKEN

import random


class ObjectDetectorInput(BaseModel):
    image: str = Field(description="The url of the image where you want to detect or mark objects.")


class ObjectDetector(BaseTool):
    name: str = "object_detector"
    description: str = "Detect objects in a given image."
    args_schema: Type[BaseModel] = ObjectDetectorInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        client = InferenceClient(token=HF_TOKEN)
        image_obj = Image.open(image)
        predicted = client.object_detection(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        draw = ImageDraw.Draw(image_obj)
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=6)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        return image_obj

    async def _arun(self, image: str) -> Image.Image:
        """Use the tool."""
        client = AsyncInferenceClient(token=HF_TOKEN)
        image_obj = Image.open(image)
        predicted = await client.object_detection(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        draw = ImageDraw.Draw(image_obj)
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=4)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        return image_obj
