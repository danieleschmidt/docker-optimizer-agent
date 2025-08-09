FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl wget
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
