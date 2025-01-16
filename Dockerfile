FROM python:3.10

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pandoc \
    qtbase5-dev \
    qttools5-dev-tools \
    qt5-qmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    qtchooser \
    build-essential \
    libgl1-mesa-dev \
    wget

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyQt6 (without strict version constraint)
RUN pip install "PyQt6>=6.7" --only-binary :all:

# Debug Qt and qmake
RUN which qmake && qmake --version

# Create the model directory
RUN mkdir -p /app/classifiers/model

# Download the model from LRZ
RUN wget --no-verbose --output-document=/app/classifiers/model/model.tar.gz \
    "https://syncandshare.lrz.de/getlink/fiRQrzQkL94w4DWzgSNPX/model"

# Extract the model (assuming it is a tar.gz file)
RUN tar -xzf /app/classifiers/model/model.tar.gz -C /app/classifiers/model && \
    rm /app/classifiers/model/model.tar.gz

# Copy project files
COPY . /app

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command
CMD ["python", "main.py"]