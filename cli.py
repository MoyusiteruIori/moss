from moss.moss import Moss
from moss.log import setup_logging
from moss.task_executor import TaskExecutor
from moss.task_planner import Plan
from moss.tools.background_eraser import BackgroundEraser
from moss.tools.canny_detector import CannyDetector
from moss.tools.depth_detector import DepthDetector
from moss.tools.chitchat import ChitChat
from moss.tools.image_generator import ImageGenerator
from moss.tools.image_qa import ImageQA
from moss.tools.image_segmenter import ImageSegmenter
from moss.tools.image_to_image_generator import ImageToImageGenerator
from moss.tools.object_detector import ObjectDetector
from moss.tools.object_replacer import ObjectReplacer
from moss.tools.pose_detector import PoseDetector
from moss.tools.sketch_refiner import SketchRefiner
from moss.tools.speech_transcriber import SpeechTranscriber
from moss.tools.summarizer import Summarizer
from moss.tools.text_reader import TextReader
from moss.tools.translator import Translator
from langchain_openai import ChatOpenAI
import time


setup_logging()


if __name__ == "__main__":

    tools = [
        BackgroundEraser(),
        CannyDetector(),
        DepthDetector(),
        ImageGenerator(),
        ImageQA(),
        ImageSegmenter(),
        ImageToImageGenerator(),
        ObjectDetector(),
        ObjectReplacer(),
        PoseDetector(),
        SketchRefiner(),
        SpeechTranscriber(),
        TextReader(),
        Translator(),
        Summarizer(),
        ChitChat(),
    ]

    agent = Moss(llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), tools=tools, verbose=False)

    execution = None
    while True:
        message = input("[ User ]: ")
        if message == "exit":
            break

        print("[ Assistant ]:")
        for res in agent.stream_with_executor_thread(
            message, "" if execution is None else str(execution)
        ):
            if isinstance(res, Plan):
                for c in f"- Planning:\n{str(res)}\n\n":
                    print(c, end='', flush=True)
                    time.sleep(0.05)
            elif isinstance(res, TaskExecutor):
                execution = res
                for c in f"- Execution:\n{str(res)}\n\n":
                    print(c, end='', flush=True)
                    time.sleep(0.05)
            elif isinstance(res, str):
                for c in f"- Final Response:\n{res}\n":
                    print(c, end='', flush=True)
                    time.sleep(0.05)
