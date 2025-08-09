FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3 pip
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
