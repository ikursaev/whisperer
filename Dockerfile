FROM python:3.11
RUN apt-get update
RUN apt-get install ffmpeg -y
WORKDIR /opt/whisperer/
COPY ./pyproject.toml /opt/whisperer/
RUN pip install .
COPY . /opt/whisperer/
# ENTRYPOINT ["python", "app.py"]