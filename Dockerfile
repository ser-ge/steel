FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /tf/home/
COPY Pipfile* /tf/home/
RUN pipenv lock --requirements > requirements.txt

COPY requirements.txt /tf/home/
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip3 install -r requirements.txt


