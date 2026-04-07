FROM python:3.8-slim
WORKDIR /anomaly

RUN apt-get update
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /anomaly

EXPOSE 5000
CMD ["python", "./app.py"]
