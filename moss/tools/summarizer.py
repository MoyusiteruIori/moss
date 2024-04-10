from typing import Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from .nlp_task_executor import load_nlp_executor


class SummarizerInput(BaseModel):
    text: str = Field(description="The text to be summarized")


class Summarizer(BaseTool):
    name: str = "summarizer"
    description: str = "A tool to summarize the given text"
    args_schema: Type[BaseModel] = SummarizerInput

    def _run(self, text: str) -> str:
        input = f"Please summarize the following text:\n---BEGIN-TEXT---\n{text}\n---END-TEXT---"
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )
        return executor.run(input=input)

    async def _arun(self, text: str) -> str:
        input = f"Please summarize the following text:\n---BEGIN-TEXT---\n{text}\n---END-TEXT---"
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )
        return await executor.arun(input=input)
