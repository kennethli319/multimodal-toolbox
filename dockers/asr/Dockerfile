# Use the official Python 3.10 image as the base image
FROM nvcr.io/nvidia/nemo:23.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

ENTRYPOINT ['bash']