FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -y python3-pip

# For development
COPY requirements.txt home/

# For Testing and production
#COPY . /home

RUN pip3 install -r home/requirements.txt

WORKDIR /home
CMD ["bash"]
