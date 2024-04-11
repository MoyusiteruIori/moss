import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import BaseTool

from pydantic.v1 import BaseModel

planner_system_template = """
#1 Task Planning Stage:
As an AI Assistant, your job is to parse user input to several tasks:
[
    {{
        "task": task_type,
        "id": task_id,
        "dep": dependency_tasks_id,
        "args": {{
            "input name": text may contain <resource-dep_id>
        }}
    }}
]. 

The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on.
The task MUST be selected from the following tools (along with tool description, input name and output type): 

{tools}. 

For example, if the user wants a picture, you should check if there is an image_generator tool or text_to_image tool, or other tools like that. If the user asks any question of a picture, you should then check if there is an image_qa tool or visual-question-answering tool.
You can deduce other cases just like these. Don't be too rigid.

There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request.
Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks.
If the user input can't be parsed, you need to reply empty JSON [].

Here are some examples for you:

------EXAMPLE-1-------
Human: Give you some pictures of sheep e1.jpg, e2.png, e3.jpg, help me count the number of sheep?
AI: [{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{ "images": ["e1.jpg", "e2.jpg", "e3.jpg"], "question": "How many sheep in total are there in these pictures?"}}}}]
---END-OF-EXAMPLE-1---

------EXAMPLE-2-------
Human: Give you a picture p1.jpg, describe it in detail and write a poem about it.
AI: [{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{"images": ["p1.jpg"], "question": "describe the image in detail and write a poem about it."}}}}]
---END-OF-EXAMPLE-2---

------EXAMPLE-3-------
Human: Listen to the description in a.mp3, and generate an image based on it. Finally write a poem about the image you generated.
AI: [{{"task": "speech_transcriber", "id": 0, "dep": [-1], "args": {{"audio": "a.mp3"}}}}, {{"task": "image_generator", "id": 1, "dep": [0], "args": {{"prompt": "<resource-0>"}}}}, {{"task": "image_qa", "id": 1, "dep": [1], "args": {{"images": ["<resource-1>"], "question": "Write a poem about this image"}}}}]
---END-OF-EXAMPLE-3---

------EXAMPLE-4-------
Human: Hi, what's your name?
AI: [{{"task": "chitchat", "id": 0, "dep": [-1], "args": {{"text": "Hi, what's your name?"}}}}]
---END-OF-EXAMPLE-4---

The user input may contain some pronouns like "the image you just generated" or "the audio I just told you". You should try to resolve these pronouns and replace them with the corresponding filename or text based on the context.
Every time you give a reply, the task list MUST start for id 0. This is import. It means you should try your best to resolve and replace pronouns based on the context, instead of introducing dependencies for a task.
If you find it impossible to resolve pronouns in user input, also reply an empty JSON [].

Here are also examples for you:

------EXAMPLE-5-------
Human: Generate an image of a white cat.
AI: [{{"task": "image_generator", "id": 0, "dep": [-1], "args": {{"prompt": "A white cat"}}}}]
Human: Previous tasks:
status: completed
result: 03e9d8.jpg
Current input: Describe the image you just generated in detail, then read it to me.
AI: [{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{"images": ["03e9d8.jpg"], "question": "describe the image in detail."}}}}, {{"task": "text_reader", "id": 1, "dep": [0], "args": {{"text": "<resource-0>"}}}}]
---END-OF-EXAMPLE-5---

------EXAMPLE-6-------
Human: Given an image ['cat.jpg'], please describe it in one sentence.
AI: [{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{"images": ["cat.jpg"], "question": "Describe the image in one sentence."}}}}]
Human: Previous tasks:
status: completed
result: A white cat with yellow eyes walking near a sofa.
Current input: Translate the description in to Chinese
AI: [{{"task": "translator", "id": 0, "dep": [-1], "args": {{"text": "A white cat with yellow eyes walking near a sofa.", "source_language": "English", "target_language": "Chinese"}}}}]
---END-OF-EXAMPLE-6---


{chat_history}"""

DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "Give you some pictures of sheep e1.jpg, e2.png, e3.jpg, help me count the number of sheep",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{ "images": ["e1.jpg", "e2.png", "e3.jpg"], "question": "How many sheep in total are there in these pictures?"}}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "Give you a picture p1.jpg, describe it in detail and write a poem about it.",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "image_qa", "id": 0, "dep": [-1], "args": {{"images": ["p1.jpg"], "question": "describe the image in detail and write a poem about it."}}}}]',
    },
    {
        "role": "user",
        "content": "Listen to the description in a.mp3, and generate an image based on it. Finally write a poem about the image you generated.",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "speech_transcriber", "id": 0, "dep": [-1], "args": {{"audio": "a.mp3"}}}}, {{"task": "image_generator", "id": 1, "dep": [0], "args": {{"prompt": "<resource-0>"}}}}, {{"task": "image_qa", "id": 1, "dep": [1], "args": {{"images": ["<resource-1>"], "question": "Write a poem about this image"}}}}]',
    },
]


class TaskPlaningChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        demos: List[Dict] = [],
        verbose: bool = True,
    ) -> LLMChain:
        """Get the response parser."""
        system_template = planner_system_template
        human_template = """{input}."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        demo_messages: List[
            Union[HumanMessagePromptTemplate, AIMessagePromptTemplate]
        ] = []
        for demo in demos:
            if demo["role"] == "user":
                demo_messages.append(
                    HumanMessagePromptTemplate.from_template(demo["content"])
                )
            else:
                demo_messages.append(
                    AIMessagePromptTemplate.from_template(demo["content"])
                )

        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose, memory=ConversationBufferMemory(memory_key="chat_history", input_key="input"))


class Step:
    """A step in the plan."""

    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool

    def __str__(self) -> str:
        return f"[{self.id}: {self.task} <- {self.args}]"


class Plan:
    """A plan to execute."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        return str([str(step) for step in self.steps])

    def __repr__(self) -> str:
        return str(self)


class BasePlanner(BaseModel):
    """Base class for a planner."""

    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decide what to do."""


class PlanningOutputParser(BaseModel):
    """Parses the output of the planning stage."""

    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        """Parse the output of the planning stage.

        Args:
            text: The output of the planning stage.
            hf_tools: The tools available.

        Returns:
            The plan.
        """
        steps = []
        for v in json.loads(text):
            choose_tool = None
            for tool in hf_tools:
                if tool.name == v["task"]:
                    choose_tool = tool
                    break
            if choose_tool:
                steps.append(Step(v["task"], v["id"], v["dep"], v["args"], tool))
        return Plan(steps=steps)


class TaskPlanner(BasePlanner):
    """Planner for tasks."""

    llm_chain: LLMChain
    output_parser: PlanningOutputParser
    stop: Optional[List] = None

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decided what to do."""
        inputs["tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        print("plan:", llm_response)
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])


def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    """Load the chat planner."""

    llm_chain = TaskPlaningChain.from_llm(llm)
    return TaskPlanner(llm_chain=llm_chain, output_parser=PlanningOutputParser())
