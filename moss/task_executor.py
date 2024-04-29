import copy
import uuid
from typing import Dict, List
import traceback
import numpy as np
from langchain.tools.base import BaseTool

from .task_planner import Plan

import logging


logger = logging.getLogger(__name__)


class Task:
    """Task to be executed."""

    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: BaseTool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool
        self.status = "pending"
        self.message = ""
        self.result = ""

    def __str__(self) -> str:
        return f"{self.task}({self.args})"

    def save_product(self) -> None:
        import cv2

        if self.task == "video_generator":
            # ndarray to video
            product = np.array(self.product)
            nframe, height, width, _ = product.shape
            video_filename = uuid.uuid4().hex[:6] + ".mp4"
            fps = 30  # Frames per second
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            for frame in self.product:
                video_out.write(frame)
            video_out.release()
            self.result = video_filename
        elif self.task in [
            "background_eraser",
            "image_generator",
            "image_segmenter",
            "object_detector",
            "object_replacer",
            "sketch_refiner"
        ]:
            # PIL.Image to image
            filename = uuid.uuid4().hex[:6] + ".png"
            self.product.save(filename)  # type: ignore
            self.result = filename
        elif self.task == "text_reader":
            # bytes to audio
            audio_filename = uuid.uuid4().hex[:6] + ".mp3"
            with open(audio_filename, "wb") as audio_file:
                audio_file.write(self.product)
            self.result = audio_filename

    def completed(self) -> bool:
        return self.status == "completed"

    def failed(self) -> bool:
        return self.status == "failed"

    def pending(self) -> bool:
        return self.status == "pending"

    def run(self) -> str:
        try:
            new_args = copy.deepcopy(self.args)
            if self.task in [
                "background_eraser",
                "image_generator",
                "image_segmenter",
                "object_detector",
                "object_replacer",
                "sketch_refiner",
                "text_reader",
                "video_generator"
            ]:
                self.product = self.tool._run(**new_args)
            else:
                self.result = self.tool._run(**new_args)
            self.status = "completed"
            self.save_product()
        except Exception as e:
            traceback.print_exc()
            self.status = "failed"
            self.message = f"{type(e).__name__}: {str(e)}"

        return self.result


class TaskExecutor:
    """Load tools and execute tasks."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.tasks: List[Task] = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task

    def completed(self) -> bool:
        return all(task.completed() for task in self.tasks)

    def failed(self) -> bool:
        return any(task.failed() for task in self.tasks)

    def pending(self) -> bool:
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task: Task) -> bool:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            if dep_task.failed() or dep_task.pending():
                return False
        return True

    def update_args(self, task: Task) -> None:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if isinstance(v, str):
                    if f"<resource-{dep_id}>" in v:
                        task.args[k] = task.args[k].replace(
                            f"<resource-{dep_id}>", dep_task.result
                        )
                elif isinstance(v, list):
                    for idx, arg in enumerate(v):
                        if f"<resource-{dep_id}>" in arg:
                            v[idx] = arg.replace(
                                f"<resource-{dep_id}>", dep_task.result
                            )

    def run(self) -> str:
        for task in self.tasks:
            logger.debug(f"running {task}")  # noqa: T201
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"
        return self.status

    def __str__(self) -> str:
        result = ""
        for task in self.tasks:
            result += f"task: {task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def describe(self) -> str:
        return self.__str__()
