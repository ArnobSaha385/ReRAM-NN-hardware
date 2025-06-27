# Start from a base image with Miniconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your entire project into the container
COPY . .

# Create the environment from the YAML file
RUN conda env create -f environment.yml

# Make sure the environment is activated by default
SHELL ["conda", "run", "-n", "reram", "/bin/bash", "-c"]

# Run inference.py by default using the created environment
CMD ["conda", "run", "-n", "reram", "python", "inference.py"]
