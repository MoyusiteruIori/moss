# Moss

Moss is my graduation project based on [HuggingGPT](https://github.com/microsoft/JARVIS). It is a system based on large language model agents designed to tackle complex artificial intelligence tasks, powerd by LangChain.


🏗️ Road Work Ahead

<a href="http://139.9.193.25:19514/images/intro.mp4">
  <img src="http://139.9.193.25:19514/images/moss-intro-cover.png" alt="moss-intro-video" width="100%"/>
</a>

</br>

Moss presents a collaborative system comprising LLM agents serving as controllers and multiple expert tools acting as collaborative executors. The workflow encompasses:

1. Task Planning: Employing LLMs to analyze user requests to comprehend their intentions and break them down into feasible solvable tasks.
2. Tool Selection: In order to address the planned tasks, LLM chooses appropriate expert tools considering their descriptions.
3. Task Execution: LLM invokes and executes each selected tool, subsequently delivering the results.
4. Response Generation: Ultimately, LLM integrates the predictions made by all tools and generates appropriate responses in friendly natural language.

## Installation

- Clone this repository

```shell
$ git clone https://github.com/MoyusiteruIori/moss.git
$ cd moss
```

- Create configuration file from template

```shell
$ cp .env.example .env
```

Replace `OPENAI_API_KEY`, `HF_TOKEN` and `SD_TOKEN` in `.env` file with your own keys/tokens. If you don't know what these are, see [openai-api](https://openai.com/blog/openai-api), [huggingface](https://huggingface.co/) and [stability.ai](https://stability.ai/).

### Install with Docker (recommended)

<b>Note</b>: Only Linux systems are guaranteed good support as the host machine for this Docker image. Ubuntu 22.04 is recommended.

Option 1: Use docker-compose (recommended):

```shell
$ docker compose up -d
# To utilize your local GPUs for certain local inferences:
# INFERENCE_MODE=HYBRID docker compose up -d
```

Option 2: Build a Docker image locally and manually start:

```shell
$ docker build -t moss .
# To utilize your local GPUs for certain local inferences:
# docker build --build-arg INFERENCE_MODE=HYBRID -t moss .
$ docker run -p 7860:7860 --name moss moss
```

### Install without Docker

Linux and macOS are supported.

- Install dependencies

```shell
conda create -n moss python=3.11
conda activate moss
pip install -r requirements.txt
```

Optional: To utilize your local GPUs for certain local inferences, you'll need some extra packages (Linux only) :

```shell
pip install accelerate==0.29.3 diffusers==0.27.2 controlnet-aux==0.0.8 transformers==4.40.1
```

- Run gradio demo

```shell
python gradio_demo.py
```

- Run command line version

```shell
python cli.py
```

You can then access `localhost:7860` to use moss.

## System Demo

Demo video for my final presentation!

<a href="http://139.9.193.25:19514/images/grad-pre.mp4">
  <img src="http://139.9.193.25:19514/images/moss-intro-cover.png" alt="sys-demo-video" width="100%"/>
</a>

[720p version](http://139.9.193.25:19514/images/grad-pre-720.mp4)

## License

[MIT](LICENSE)

## Credits

- [HuggingGPT](http://arxiv.org/abs/2303.17580)