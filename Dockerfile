FROM python:3.11
RUN apt-get update
RUN apt-get install ffmpeg -y
WORKDIR /opt/whisperer/
COPY ./requirements.txt /opt/whisperer/
RUN pip install -r requirements.txt
COPY . /opt/whisperer/
ENTRYPOINT ["python", "app.py"]