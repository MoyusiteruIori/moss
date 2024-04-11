from typing import Any, Callable, Generator, List, Optional, Tuple, TypedDict, TypeVar
from moss.moss import Moss
from moss.task_executor import TaskExecutor
from moss.task_planner import Plan
from moss.tools.chitchat import ChitChat
from moss.tools.image_generator import ImageGenerator
from moss.tools.image_qa import ImageQA
from moss.tools.speech_transcriber import SpeechTranscriber
from moss.tools.summarizer import Summarizer
from moss.tools.text_reader import TextReader
from moss.tools.translator import Translator
from langchain_openai import ChatOpenAI

import os
import random
import shutil
import time

import gradio as gr  # type: ignore


def load_agent() -> Moss:
    return Moss(llm=ChatOpenAI(temperature=0, model="gpt-4"), tools=tools)


tools = [
    ImageGenerator(),
    ImageQA(),
    TextReader(),
    SpeechTranscriber(),
    Translator(),
    Summarizer(),
    ChitChat(),
]

agent = load_agent()


class UserMessageType(TypedDict):
    text: str
    files: List[str]


HistoryType = List[List[Tuple[str] | str | None]]


def clear_workspace():
    global agent
    global previous_execution
    agent = load_agent()
    previous_execution = None
    cwd = os.getcwd()
    for filename in os.listdir(cwd):
        if filename.endswith((".png", ".jpg", ".mp3")):
            path = os.path.join(cwd, filename)
            os.remove(path)


def find_unreplied_user_message(history: HistoryType) -> UserMessageType:
    latest_user_message: UserMessageType = {"files": [], "text": ""}
    for msg in reversed(history):
        if msg[0] is None:
            break
        if isinstance(msg[0], tuple):
            latest_user_message["files"].insert(0, msg[0][0])
        elif isinstance(msg[0], str):
            latest_user_message["text"] = msg[0]
    return latest_user_message


def no_reply_before(history: HistoryType) -> bool:
    for msg in history:
        if msg[0] is None:
            return False
    return True


def is_file(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def format_plan_markdown(plan: Plan) -> str:
    result = "## Task Planner:\n"
    result += "The request you submitted has been processed by the task planner and has been scheduled into the following tasks:\n\n"
    for index, step in enumerate(plan.steps):
        result += f"{index + 1}. **{step.task}** ( {step.args} )\n"
    result += "\n Now execution starts. Please wait...."
    return result


def format_execution_markdown(execution: TaskExecutor) -> List[str]:
    s = "## Task Executor:\n"
    s += "The task executor has executed the tasks. Here's the results:\n\n"
    result: List[str] = [s]
    for index, task in enumerate(execution.tasks):
        s = f"{index + 1}. **{task.task}** - **{task.status}**  "
        if task.status != "completed":
            s += f"message: {task.message}"
        else:
            if not is_file(task.result):
                s += f"result: {task.result}"
        result.append(s)
        if is_file(task.result):
            result.append(task.result)
    return result


def format_response_markdown(response: str) -> str:
    result = "## Response Generator:\n"
    result += "The response generator summarized the execution results, "
    result += f"Here is the final answer:\n\n {response}"
    return result


T = TypeVar("T")


def resp_generator(text: T, formatter: Callable[[T], str]) -> Generator[str, Any, None]:
    formatted_text = formatter(text)
    for c in formatted_text:
        yield c
        time.sleep(random.uniform(0, 0.03))


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history: HistoryType, message: UserMessageType):
    for x in message["files"]:
        history.append([(x,), None])
    if message["text"] is not None:
        history.append([message["text"], None])
    return history, gr.MultimodalTextbox(value=None, interactive=False)


previous_execution: Optional[TaskExecutor] = None


def bot(history: HistoryType):
    global previous_execution
    global agent
    if no_reply_before(history):
        agent = load_agent()
        previous_execution = None
    unreplied_user_message = find_unreplied_user_message(history)
    files: List[str] = []
    for file in unreplied_user_message["files"]:
        filename = os.path.basename(file)
        shutil.copyfile(file, f"./{filename}")
        files.append(filename)

    formatted_input = f"Given these files: {files}, " if len(files) > 0 else ""
    formatted_input += f"{unreplied_user_message['text']}"
    for res in agent.stream_with_executor_thread(
        formatted_input, "" if previous_execution is None else str(previous_execution)
    ):
        if isinstance(res, Plan):
            history.append([None, ""])
            for c in resp_generator(res, format_plan_markdown):
                history[-1][1] += c  # type: ignore
                yield history
        elif isinstance(res, TaskExecutor):
            previous_execution = res
            reply_list = format_execution_markdown(res)
            for reply in reply_list:
                if is_file(reply):
                    history.append([None, (reply,)])
                    yield history
                else:
                    history.append([None, ""])
                    for c in resp_generator(reply, lambda s: s):
                        history[-1][1] += c  # type: ignore
                        yield history
        elif isinstance(res, str):
            history.append([None, ""])
            for c in resp_generator(res, format_response_markdown):
                history[-1][1] += c  # type: ignore
                yield history


if __name__ == "__main__":

    theme = gr.themes.Glass(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.zinc,
        radius_size=gr.themes.sizes.radius_lg,
    )

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("<h1><center> 🚀 Moss Gradio Demo</center></h1>")
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=("assets/user_avatar.jpg", "assets/bot_avatar.png"),
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image", "audio"],
            placeholder="Enter message or upload file...",
            show_label=False,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        clear_btn = gr.ClearButton([chat_input, chatbot]).click(fn=clear_workspace)

        chatbot.like(print_like_dislike, None, None)

        examples = gr.Examples(
            [
                {
                    "text": "Describe these images in detail. Tell me every thing that you see.",
                    "files": ["examples/black_cat.png", "examples/white_cat.png"],
                },
                {
                    "text": "Generate an image of a knight riding a dragon, flying in the sky. Then write a poem about the image, and dub it. Finally translate the poem into Chinese.",
                    "files": [],
                },
                {
                    "text": "Generate two images of cats, one white and the other black, and provide a comparative description for these two images.",
                    "files": [],
                },
                {
                    "text": "Listen to the speech in this audio file, summarize the content, and generate an image based on it.",
                    "files": ["examples/poem.mp3"],
                },
            ],
            chat_input,
        )

    demo.queue()
    demo.launch()
