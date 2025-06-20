FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Let uv handle Python version management
ENV UV_PYTHON="3.11"

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync

# PyTorch official image should have compatible CUDA + cuDNN versions

# Create entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directories for shared volumes
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["./entrypoint.sh"]