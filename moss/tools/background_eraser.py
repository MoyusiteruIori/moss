from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field
from ..configs import SD_TOKEN

import io
import requests


class BackgroundEraserInput(BaseModel):
    image: str = Field(description="The url of the image to remove background")


class BackgroundEraser(BaseTool):
    name: str = "background_eraser"
    description: str = "Remove the background of an image"
    args_schema: Type[BaseModel] = BackgroundEraserInput

    def _run(self, image: str) -> Image.Image:
        """Use the tool."""
        headers = {
            "authorization": f"Bearer {SD_TOKEN}",
            "accept": "image/*"
        }
        files = { "image": open(image, "rb") }
        data = { "output_format": "png" }
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/remove-background",
            headers=headers,
            files=files,
            data=data
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise Exception(str(response.json()))



    async def _arun(self, image: str) -> Image.Image:
        """Async run is not implemented yet. Call sync run instead."""
        return self._run(image)