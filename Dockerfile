FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file (if you have one) and install dependencies
COPY req.txt ./
RUN pip install --no-cache-dir -r req.txt

# Copy the entire codebase into the container
COPY . .

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

