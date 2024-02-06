# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

RUN pip install --upgrade pip

RUN pip install Cython

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

ENTRYPOINT ['bash']