FROM python:3.11-alpine
RUN apk --no-cache add gcc musl-dev
RUN apk add  --no-cache ffmpeg
COPY . /opt/whisperer/
WORKDIR /opt/whisperer/
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "app.py"]