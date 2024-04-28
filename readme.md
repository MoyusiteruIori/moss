# Moss

My graduation project based on [HuggingGPT](https://github.com/microsoft/JARVIS)

🏗️ Road Work Ahead

<a href="http://139.9.193.25:19514/images/intro.mp4">
  <img src="http://139.9.193.25:19514/images/moss-intro-cover.png" alt="moss-intro-video" width="100%"/>
</a>

## Installation

### Install with Docker

TODO

### Install without Docker

- Clone this repository

```shell
git clone https://github.com/MoyusiteruIori/moss.git
cd moss
```

- Create configuration file from template

```shell
cp .env.example .env
```

Replace `OPENAI_API_KEY`, `HF_TOKEN` and `SD_TOKEN` with your own. If you don't know what these are, see [openai-api](https://openai.com/blog/openai-api), [huggingface](https://huggingface.co/) and [stability.ai](https://stability.ai/).

- Install dependencies

```shell
conda create -n moss python=3.11
conda activate moss
pip install -r requirements.txt
```

- Run gradio demo

```shell
python gradio_demo.py
```

- Run command line version

```shell
python cli.py
```