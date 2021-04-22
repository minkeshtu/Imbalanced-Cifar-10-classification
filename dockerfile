####### Docker Build
##### docker build -t <image name> .

####### Docker run
##### docker run --runtime=nvidia -it --name <container_name> <image_name>

FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
####### For cuda 10.1 ->
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update \
        && apt-get --yes install \
        wget \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cudatoolkit -c pytorch
######## For cuda 10.1 ->
#RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


RUN pip install -U scikit-learn
RUN pip install Pillow
RUN pip install tensorboard
RUN pip install matplotlib
RUN pip install pandas
RUN pip install seaborn
RUN pip install jupyterlab
RUN pip install torch-summary
RUN pip install pyyaml
RUN pip install python-docx
RUN pip install Cython

COPY . /home

WORKDIR /home