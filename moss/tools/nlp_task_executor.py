from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


class NLPTasksExecutorChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        template = "You are a helpful assistant who can help the user with many tasks like translation, summarization, conversation and so on.\n\n{chat_history}\nHuman: {input}"
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
        )
        return cls(
            prompt=prompt,
            llm=llm,
            verbose=verbose,
            memory=ConversationBufferMemory(
                memory_key="chat_history", input_key="input"
            ),
        )


class NLPTasksExecutor:
    """Handle NLP Tasks"""

    def __init__(self, llm_chain: LLMChain, stop: Optional[List] = None):
        self.llm_chain = llm_chain
        self.stop = stop

    def run(self, input: str, callbacks: Callbacks = None, **kwargs: Any) -> str:
        llm_response = self.llm_chain.run(
            input=input, stop=self.stop, callbacks=callbacks
        )
        return llm_response
    
    async def arun(self, input: str, callbacks: Callbacks = None, **kwargs: Any) -> str:
        llm_response = await self.llm_chain.arun(
            input=input, stop=self.stop, callbacks=callbacks
        )
        return llm_response


def load_nlp_executor(llm: BaseLanguageModel) -> NLPTasksExecutor:
    """Load the ResponseGenerator."""

    llm_chain = NLPTasksExecutorChain.from_llm(llm)
    return NLPTasksExecutor(
        llm_chain=llm_chain,
    )
