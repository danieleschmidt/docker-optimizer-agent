# Dockerfile with multiple security issues for testing
FROM ubuntu:latest
USER root
RUN apt-get update
RUN apt-get install -y curl wget sudo
COPY --chown=root:root secrets.txt /tmp/
EXPOSE 22 3389
ENV PASSWORD=admin123
WORKDIR /
CMD ["sh", "-c", "chmod 777 /tmp && while true; do sleep 1; done"]