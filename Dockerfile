###########
## Usage ##
###########

# sp-society-camera-model-identification should be in the $CODE directory.

# alias andres="docker run --runtime=nvidia --init -it --rm \
#               --ipc=host \
#               -v $CODE:/code -v $DATA:/data \
#               -w=/code/sp-society-camera-model-identification \
#               mwksmith/cam:andres"

# When you get into the container enter "sa" to activate the conda environment,
# then create sym links to your data folders according to the globals set in train.py,
# then run train.py.

# Direct questions here: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/48293#274879

# Mini-tutorial on docker run options: https://github.com/MattKleinsmith/dockerfiles/tree/master/fastai#explanation

###########

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# To allow unicode characters in the terminal
ENV LANG C.UTF-8

###########
## Tools ##
###########

RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    vim \
    git \
    unzip

##############
## Anaconda ##
##############

RUN apt-get update --fix-missing && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

################
## Tensorflow ##
################

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip

RUN conda create -n tf python=3.5 anaconda

RUN /bin/bash -c "\
    source activate tf && \
    pip install --ignore-installed --upgrade \
        https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl"

###########
## Keras ##
###########

RUN /bin/bash -c "source activate tf && \
    pip install \
        keras"

#########################
## Andres dependencies ##
#########################

RUN /bin/bash -c "source activate tf && \
    pip install \
        numpngw \
        tqdm \
        jpeg4py \
        opencv-python \
        conditional"

RUN apt-get install -y libturbojpeg

# Add sym links to:
#   train
#   test
#   models

###########
## Other ##
###########

# TODO: MKL

RUN echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc

RUN echo "alias sa=\"source activate tf\"" >> ~/.bashrc

CMD bash
