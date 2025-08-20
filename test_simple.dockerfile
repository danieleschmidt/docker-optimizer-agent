FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]