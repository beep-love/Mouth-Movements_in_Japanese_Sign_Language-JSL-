# Use an official NVIDIA CUDA runtime with cuDNN and Ubuntu as a base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for Python packages like OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*  # Cleanup to reduce image size

# Upgrade pip to ensure we have the latest version for installing Python packages
RUN pip install --no-cache-dir --upgrade pip

# Install required Python libraries
RUN pip install --no-cache-dir \
    opencv-python-headless \
    torch torchvision \
    pympi-ling \
    numpy \
    pandas \
    matplotlib \
    Pillow \
    onnx \
    onnxruntime-gpu \
    insightface \
    mediapipe \
    seaborn

# Set the working directory in the container
WORKDIR /usr/src/app

# Optional: Define the default command to run when starting the container
# CMD ["python3"]