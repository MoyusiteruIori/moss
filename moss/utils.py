from typing import Any, Dict
import base64
import requests
import aiohttp
import re
from io import BytesIO
from PIL import Image


def base64_to_image(base64_str: str, image_path=None) -> Image.Image:
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def send_openai_post(
    api_endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(api_endpoint, headers=headers, json=payload)
    return response


async def async_send_openai_post(
    api_endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
) -> aiohttp.ClientResponse:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_endpoint, json=payload, headers=headers, 
        ) as resp:
            return resp
