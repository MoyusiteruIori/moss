from typing import Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from .nlp_task_executor import load_nlp_executor


class TranslatorInput(BaseModel):
    text: str = Field(description="The text to be translated")
    source_language: str = Field(description="The source language of the translation")
    target_language: str = Field(description="The target language of the translation")


class Translator(BaseTool):
    name: str = "translator"
    description: str = "Translate a piece of text from the source language to the target language"
    args_schema: Type[BaseModel] = TranslatorInput

    def _run(self, text: str, source_language: str, target_language: str) -> str:
        input = f"Please translate the following text from {source_language} into {target_language}:\n---BEGIN-TEXT---\n{text}\n---END-TEXT---"
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )
        return executor.run(input=input)

    async def _arun(self, text: str, source_language: str, target_language: str) -> str:
        input = f"Please translate the following text from {source_language} into {target_language}:\n---BEGIN-TEXT---\n{text}\n---END-TEXT---"
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )
        return await executor.arun(input=input)
