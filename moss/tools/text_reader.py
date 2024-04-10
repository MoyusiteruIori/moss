from typing import Type
from langchain.tools import BaseTool
from openai import OpenAI, AsyncOpenAI
from pydantic.v1 import BaseModel, Field
from ..configs import OPENAI_API_KEY, OPENAI_API_BASE


class TextReaderInput(BaseModel):
    text: str = Field(description="The text used to generate a piece of audio.")


class TextReader(BaseTool):
    name: str = "text_reader"
    description: str = "Generate a speech audio based on the text."
    args_schema: Type[BaseModel] = TextReaderInput

    def _run(self, text: str) -> bytes:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="nova", input=text, response_format="mp3"
        ) as response:
            # response.stream_to_file("test.mp3")
            return response.read()

    async def _arun(self, text: str) -> bytes:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        async with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="nova", input=text, response_format="mp3"
        ) as response:
            return await response.read()
