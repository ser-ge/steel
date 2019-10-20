FROM tensorflow/tensorflow:1.15.0rc1-gpu-py3-jupyter

WORKDIR /tf/home/

RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip3 install pandas opencv-python seaborn sklearn keras


