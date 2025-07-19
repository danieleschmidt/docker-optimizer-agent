FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y vim
COPY . /app
WORKDIR /app
EXPOSE 8080