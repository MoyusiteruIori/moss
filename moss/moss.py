from typing import Any, Generator, Sequence, Tuple

from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from .repsonse_generator import (
    load_response_generator,
)
from .task_executor import TaskExecutor, TaskExecutor
from .task_planner import load_chat_planner, Plan


class Moss:

    def __init__(self, llm: BaseLanguageModel, tools: Sequence[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor: TaskExecutor

    def run(
        self, input: str, previous_result: str = ""
    ) -> Tuple[Plan, TaskExecutor, str]:
        if previous_result == "":
            real_input = f"\n- Current Input: {input}"
        else:
            real_input = f"\n- Previous tasks result: {previous_result}\n- Current Input: {input}"
        plan = self.chat_planner.plan(
            inputs={
                "input": real_input,
                "hf_tools": self.tools,
            }
        )
        print(f"Task planning result: {str(plan)}\n")
        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()
        response = self.response_generator.generate(
            {"input": input, "task_execution": self.task_executor}
        )
        return plan, self.task_executor, response

    def stream(
        self, input: str, previous_result: str = ""
    ) -> Generator[Plan | TaskExecutor | str, Any, None]:
        if previous_result == "":
            real_input = f"\n- Current Input: {input}"
        else:
            real_input = f"\n- Previous tasks result: {previous_result}\n- Current Input: {input}"
        plan = self.chat_planner.plan(
            inputs={
                "input": real_input,
                "hf_tools": self.tools,
            }
        )
        print(f"Task planning result: {str(plan)}\n")
        yield plan

        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()
        yield self.task_executor

        response = self.response_generator.generate(
            {"input": input, "task_execution": self.task_executor}
        )
        yield response
