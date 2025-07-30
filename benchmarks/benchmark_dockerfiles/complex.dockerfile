# Complex benchmark Dockerfile for testing optimization of large files
FROM node:18-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY python-requirements.txt .
RUN pip3 install -r python-requirements.txt

# Install Node.js dependencies  
COPY package*.json ./
RUN npm ci --only=production

# Install global tools
RUN npm install -g typescript
RUN npm install -g @vue/cli
RUN npm install -g create-react-app

# Copy application code
COPY . /app
WORKDIR /app

# Build steps
RUN npm run build
RUN python3 setup.py build

# Runtime configuration
EXPOSE 3000 8000
VOLUME ["/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["npm", "start"]