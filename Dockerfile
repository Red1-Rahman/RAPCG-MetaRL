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

# Install dashboard dependency
RUN pip install --no-cache-dir streamlit

# Copy project files
COPY . .

# Install gym-pcgrl
RUN cd gym-pcgrl && pip install -e . && cd ..

# Create necessary directories
RUN mkdir -p logs checkpoints generated_levels dashboard

# Set Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Streamlit config — disable telemetry, set port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose Streamlit port
EXPOSE 8501

# Default: launch dashboard
CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]