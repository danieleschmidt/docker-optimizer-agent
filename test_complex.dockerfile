FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y git
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["python3", "app.py"]