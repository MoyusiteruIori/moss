from typing import Any, Generator, Sequence, Tuple
from queue import Queue
from threading import Thread

from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from .repsonse_generator import (
    load_response_generator,
)
from .task_executor import TaskExecutor, TaskExecutor
from .task_planner import load_chat_planner, Plan


class Moss:

    def __init__(self, llm: BaseLanguageModel, tools: Sequence[BaseTool], verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.chat_planner = load_chat_planner(llm, verbose=verbose)
        self.response_generator = load_response_generator(llm, verbose=verbose)
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

    def stream_with_executor_thread(
        self, input: str, previous_result: str = ""
    ) -> Generator[Plan | TaskExecutor | str, Any, None]:

        q: Queue[Plan | TaskExecutor | str] = Queue()

        def executor_func():
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
            q.put(plan)

            self.task_executor = TaskExecutor(plan)
            self.task_executor.run()
            q.put(self.task_executor)

            response = self.response_generator.generate(
                {"input": input, "task_execution": self.task_executor}
            )
            q.put(response)

        executor_thread = Thread(target=executor_func)
        executor_thread.start()

        for res_type in ["plan", "execution", "response"]:
            res = q.get()
            yield res

        executor_thread.join()
