FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install gym-pcgrl
RUN cd gym-pcgrl && pip install -e . && cd ..

# Create necessary directories
RUN mkdir -p logs checkpoints generated_levels

# Set Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Default command
CMD ["python", "train.py", "--help"]
