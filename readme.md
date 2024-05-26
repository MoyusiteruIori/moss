# Moss

My graduation project based on [HuggingGPT](https://github.com/microsoft/JARVIS)

🏗️ Road Work Ahead

<a href="http://139.9.193.25:19514/images/intro.mp4">
  <img src="http://139.9.193.25:19514/images/moss-intro-cover.png" alt="moss-intro-video" width="100%"/>
</a>

## Installation

- Clone this repository

```shell
$ git clone https://github.com/MoyusiteruIori/moss.git
$ cd moss
$ cp .env.example .env
```

- Create configuration file from template

Replace `OPENAI_API_KEY`, `HF_TOKEN` and `SD_TOKEN` with your own. If you don't know what these are, see [openai-api](https://openai.com/blog/openai-api), [huggingface](https://huggingface.co/) and [stability.ai](https://stability.ai/).

### Install with Docker

Option 1: Use docker-compose (recommended):

```shell
$ docker compose up -d
# To utilize your local GPUs for certain local inferences:
# INFERENCEMODE=HYBRID docker compose up -d
```

Option 2: Build a Docker image locally and manually start:

```shell
$ docker build -t moss .
# To utilize your local GPUs for certain local inferences:
# docker build -t --build-arg INFERENCEMODE=HYBRID moss .
$ docker run -p 7860:7860 --name moss moss
```

### Install without Docker

- Install dependencies

```shell
conda create -n moss python=3.11
conda activate moss
pip install -r requirements.txt
```

Optional: To utilize your local GPUs for certain local inferences, you'll need some extra packages:

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