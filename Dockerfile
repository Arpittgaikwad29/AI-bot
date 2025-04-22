# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies for audio and tkinter
RUN apt-get update && apt-get install -y \
    python3-tk \
    espeak \
    libespeak1 \
    libportaudio2 \
    libpulse-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy your code into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the command to run your app
CMD ["python", "app.py"]
