from typing import Type
from langchain.tools import BaseTool
from openai import OpenAI, AsyncOpenAI
from pydantic.v1 import BaseModel, Field
from ..configs import OPENAI_API_KEY, OPENAI_API_BASE


class SpeechTranscriberInput(BaseModel):
    audio: str = Field(description="The url of the speech audio to be recognized")


class SpeechTranscriber(BaseTool):
    name: str = "speech_transcriber"
    description: str = "Recognize the input speech audio and generate the text"
    args_schema: Type[BaseModel] = SpeechTranscriberInput

    def _run(self, audio: str) -> str:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=open(audio, "rb")
        )
        return transcription.text

    async def _arun(self, audio: str) -> str:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        transcription = await client.audio.transcriptions.create(
            model="whisper-1", file=open(audio, "rb")
        )
        return transcription.text
