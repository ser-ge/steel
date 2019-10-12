FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /tf /home/

COPY requirements.txt /home/
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip install -r requirements.txt


