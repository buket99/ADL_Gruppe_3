# Use a base image with Python 3.12
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including Pandoc and GUI dependencies
RUN apt-get update && apt-get install -y \
    pandoc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if necessary (e.g., for a web interface)
# EXPOSE 8080

# Set the default command to run the main application
CMD ["python", "main.py"]