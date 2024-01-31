# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
