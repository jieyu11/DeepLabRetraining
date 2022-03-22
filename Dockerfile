FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# runtime or devel ?
# Install
RUN apt-get update && apt-get install -y \
    wget vim python3.6 \
    apt-utils \
    locales \
    libgtk2.0-dev \
    imagemagick \
    libgl1
#     python-tk \

RUN pip install --upgrade pip
RUN pip install numpy==1.18.5 \
    scipy==1.7.1 \
    h5py==3.5.0 \
    tqdm==4.46.0 \
    torchvision==0.7 \
    gdown \
    opencv-python==4.5.3.56

ENV WORK_ROOT /work
RUN mkdir $WORK_ROOT
WORKDIR $WORK_ROOT

COPY *.py $WORK_ROOT/
RUN python download_pretrained.py
RUN gdown 1BwLXPc0BOPN4ZA5s-rowJwIQPVMX4ene --id -O weights_deeplabv3_retrain_coco_width480_height640.pt
COPY libs_internal libs_internal
