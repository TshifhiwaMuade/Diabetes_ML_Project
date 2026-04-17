
FROM python:3.11-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# Note: Ensure 'gunicorn' is in your requirements.txt 
# or we can install it manually here:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of your app
COPY . .

# Start the app using Gunicorn
# 'web_app' refers to web_app.py
# 'server' refers to the line 'server = app.server' in your code
CMD ["gunicorn", "web_app:server", "--bind", "0.0.0.0:8080"]# 1. Use an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (This fixes your Rust/Build errors!)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app code
COPY . .

# 6. The command to run your app
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080"]
