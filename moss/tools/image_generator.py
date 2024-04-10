from typing import Type
from openai import OpenAI, AsyncOpenAI
from langchain.tools import BaseTool
from PIL import Image
from pydantic.v1 import BaseModel, Field
from ..configs import OPENAI_API_KEY, OPENAI_API_BASE
from ..exceptions import TaskExecutionError
from ..utils import base64_to_image


class ImageGeneratorInput(BaseModel):
    prompt: str = Field(description="The prompt of generating an image.")


class ImageGenerator(BaseTool):
    name: str = "image_generator"
    description: str = "Generate an image based on the prompt."
    args_schema: Type[BaseModel] = ImageGeneratorInput

    def _run(self, prompt: str) -> Image.Image:
        """Use the tool."""
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        if response.data[0].b64_json is not None:
            img = base64_to_image(response.data[0].b64_json)
            return img
        else:
            raise TaskExecutionError("Failed to generate image: b64_json is None.")

    async def _arun(self, prompt: str) -> Image.Image:
        """Use the tool asynchronously."""
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        if response.data[0].b64_json is not None:
            img = base64_to_image(response.data[0].b64_json)
            return img
        else:
            raise TaskExecutionError("Failed to generate image: b64_json is None.")
