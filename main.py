from moss.moss import Moss
from moss.tools.chitchat import ChitChat
from moss.tools.image_generator import ImageGenerator
from moss.tools.image_qa import ImageQA
from moss.tools.summarizer import Summarizer
from moss.tools.speech_transcriber import SpeechTranscriber
from moss.tools.text_reader import TextReader
from moss.tools.translator import Translator
from langchain_openai import ChatOpenAI
import os


if __name__ == "__main__":

    tools = [
        ImageGenerator(),
        ImageQA(),
        TextReader(),
        SpeechTranscriber(),
        Translator(),
        Summarizer(),
        ChitChat()
    ]

    agent = Moss(llm=ChatOpenAI(temperature=0, model="gpt-4"), tools=tools)

    execution = None
    while True:
        message = input("[ User ]: ")
        if message == "exit":
            break

        plan, execution, response = agent.run(
            message, "" if execution is None else str(execution)
        )
        print(
            f"[ Assistant ]:\n- Planning:\n{str(plan)}\n\n- Execution:\n{str(execution)}\n\n- Final Response:\n{response}"
        )
