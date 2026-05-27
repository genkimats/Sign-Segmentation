# Use Miniconda as the base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your conda environment file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Ensure the new environment is activated automatically
RUN echo "conda activate myenv" >> ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Copy the rest of the source code
COPY . .

# Default command to run when the container starts
CMD ["python", "main.py"]