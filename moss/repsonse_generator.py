from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


class ResponseGenerationChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        execution_template = "{chat_history}\n\n" "{input}"
        prompt = PromptTemplate(
            template=execution_template,
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


class ResponseGenerator:
    """Generates a response based on the input."""

    def __init__(self, llm_chain: LLMChain, stop: Optional[List] = None):
        self.llm_chain = llm_chain
        self.stop = stop

    def generate(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Given input, decided what to do."""
        real_input = f"""
The user has input a query as follows:

-----QUERY-----
{inputs["input"]}
--END-OF-QUERY--

An AI assistant has parsed the user input into several tasks and executed them. The results are as follows:

-----RESULT-----
{inputs["task_execution"]}
--END-OF-RESULT--

Please summarize the results according to the user input and generate a response.
"""
        llm_response = self.llm_chain.run(
            input=real_input, stop=self.stop, callbacks=callbacks
        )
        return llm_response


def load_response_generator(llm: BaseLanguageModel, verbose: bool = True) -> ResponseGenerator:
    """Load the ResponseGenerator."""

    llm_chain = ResponseGenerationChain.from_llm(llm, verbose=verbose)
    return ResponseGenerator(
        llm_chain=llm_chain,
    )
