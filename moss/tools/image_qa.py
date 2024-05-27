from typing import List, Type
from langchain.tools import BaseTool
from pydantic.v1 import BaseModel, Field
from ..configs import OPENAI_API_KEY, OPENAI_API_CHAT_ENDPOINT
from ..exceptions import TaskExecutionError
from ..utils import encode_image, send_openai_post, async_send_openai_post


class ImageQAInput(BaseModel):
    images: List[str] = Field(
        description="The list of urls of the images for the question-answering."
    )
    question: str = Field(description="The question about the input images.")


class ImageQA(BaseTool):
    name: str = "image_qa"
    description: str = (
        "Answer question about images. This tool is powerful, feel free to input multiple images and complex question."
    )
    args_schema: Type[BaseModel] = ImageQAInput

    def _run(self, images: List[str], question: str) -> str:
        image_message_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
            }
            for image in images
        ]
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": image_message_content
                    + [{"type": "text", "text": question}],
                }
            ],
        }
        if OPENAI_API_CHAT_ENDPOINT is None or OPENAI_API_KEY is None:
            raise TaskExecutionError("openai request error.")
        return send_openai_post(
            OPENAI_API_CHAT_ENDPOINT, OPENAI_API_KEY, payload
        ).json()["choices"][0]["message"]["content"]

    async def _arun(self, image: str, question: str) -> str:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image)}"
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        }
        if OPENAI_API_CHAT_ENDPOINT is None or OPENAI_API_KEY is None:
            raise TaskExecutionError("openai request error.")
        return (
            await (
                await async_send_openai_post(
                    OPENAI_API_CHAT_ENDPOINT, OPENAI_API_KEY, payload
                )
            ).json()
        )["choices"][0]["message"]["content"]
