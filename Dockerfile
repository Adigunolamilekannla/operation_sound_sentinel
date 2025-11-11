FROM python:3.10-slim-buster
WORKDIR /application
COPY . /application

RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install -r requirements.txt
CMD ["python", "application.py"]