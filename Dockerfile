FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /home/
COPY Pipfile* /home/
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r requirements.txt


