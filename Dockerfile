ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" 
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN apt-get -y update

WORKDIR /app
RUN cd /app
COPY ./yolov7 ./yolov7
RUN pip3 install -r ./yolov7/requirements.txt

RUN cd /app
COPY ./beit ./beit
RUN pip3 install -r ./beit/requirements.txt

RUN pip3 uninstall -y opencv-python && \
    pip3 install opencv-python-headless

WORKDIR /app
RUN cd /app
COPY ./ocr ./ocr

WORKDIR /app
RUN cd /app
COPY ./data_processing ./data_processing

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*
