# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from your repo into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir torch numpy matplotlib

# Run the inference script by default
CMD ["python", "inference.py"]
