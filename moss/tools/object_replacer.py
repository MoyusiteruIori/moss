from typing import Type
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field
from ..configs import SD_TOKEN

import io
import requests


class ObjectReplacerInput(BaseModel):
    image           : str = Field(title="image", description="The url of the image in which an object is to be replaced")
    original_object : str = Field(title="original_object", description="The object which is going to be removed in the image")
    new_object      : str = Field(title="new_object", description="The new object to add in the image")


class ObjectReplacer(BaseTool):
    name: str = "object_replacer"
    description: str = "Replace certain object in an image"
    args_schema: Type[BaseModel] = ObjectReplacerInput

    def _run(self, image: str, original_object: str, new_object: str) -> Image.Image:
        """Use the tool."""
        headers = {
            "authorization": f"Bearer {SD_TOKEN}",
            "accept": "image/*"
        }
        files = { "image": open(image, "rb") }
        data = {
            "prompt": new_object,
            "search_prompt": original_object,
            "output_format": "png"
        }
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace",
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
