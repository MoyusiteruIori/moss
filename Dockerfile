FROM python:3.11-bullseye
LABEL maintainer="1or1"
LABEL version="v1"

ARG INFERENCE_MODE
ENV INFERENCE_MODE=${INFERENCE_MODE}

WORKDIR /opt

RUN /usr/local/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain==0.1.16 langchain-openai==0.1.3 gradio python-dotenv opencv-python==4.9.0.80 toml
RUN if [ ${INFERENCE_MODE} = "HYBRID" ]; \
    then \
        /usr/local/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate==0.29.3 diffusers==0.27.2 controlnet-aux==0.0.8 transformers==4.40.1 && echo "Starting in hybrid inference mode..."; \
    else \
        echo "Starting in fully online inference mode..."; \
    fi

COPY . /opt/moss
WORKDIR /opt/moss

ENTRYPOINT ["/usr/local/bin/python", "./gradio_demo.py"]

EXPOSE 7860