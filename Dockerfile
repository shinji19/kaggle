FROM python:3.7.0

RUN pip install --upgrade pip
WORKDIR /tmp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
