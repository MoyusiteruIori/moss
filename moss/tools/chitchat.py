from typing import Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from .nlp_task_executor import load_nlp_executor


class ChitChatInput(BaseModel):
    text: str = Field(description="The chat input")


class ChitChat(BaseTool):
    name: str = "chitchat"
    description: str = "Chat with the user when the user request contains no AI tasks."
    args_schema: Type[BaseModel] = ChitChatInput

    def _run(self, text: str) -> str:
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            verbose=False
        )
        return executor.run(input=text)

    async def _arun(self, text: str) -> str:
        executor = load_nlp_executor(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            verbose=False
        )
        return await executor.arun(input=text)
