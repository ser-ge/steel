FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /home/

COPY requirements.txt /home/
RUN pip install -r requirements.txt


