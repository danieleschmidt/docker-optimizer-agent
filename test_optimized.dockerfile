🚀 Docker Optimization Results
========================================
Original Size: 110MB
Optimized Size: 110MB
Explanation: Applied 2 security improvements; Applied 1 layer optimizations

🔒 Security Fixes Applied: 2

⚡ Layer Optimizations: 1

📄 Optimized Dockerfile:
------------------------------
FROM ubuntu:22.04
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl wget && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
USER 1001:1001
CMD ["python3", "app.py"]