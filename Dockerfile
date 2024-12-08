FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file (if you have one) and install dependencies
COPY req.txt ./
RUN pip install --upgrade pip && pip install -r req.txt

# Copy the entire codebase into the container
COPY . .

CMD ["python", "app.py"]
