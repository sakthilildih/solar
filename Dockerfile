# Dockerfile for Solar Panel Detection Pipeline
FROM python:3.10-slim

# Set maintainer label
LABEL maintainer="EcoInnovators"
LABEL description="Solar Panel Detection Pipeline with Ensemble YOLO Models"

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY ["best.pt", "best (2).pt", "./"]

# Copy pipeline code
COPY pipeline.py .

# Create directories for input/output
RUN mkdir -p input output/images output/artifacts output/json

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_DEBUG=0

# Default command - run the pipeline
CMD ["python", "pipeline.py"]
