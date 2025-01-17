FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    libgl1-mesa-dev \
    unzip \
    libx11-xcb1 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyQt5 (adjust to specific modules if possible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pyqt5 \
    python3-pyqt5.qtwebengine \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app

# Create the model directory and download models
RUN mkdir -p /app/classifiers/model
COPY download_models.py /app/
RUN python /app/download_models.py

# Set the default command
CMD ["python", "main.py"]