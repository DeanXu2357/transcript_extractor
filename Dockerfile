# Multi-stage build for dependency caching
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS dependencies

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

# Copy only dependency files (changes less frequently)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies - this layer will be cached
RUN uv sync

# Final stage
FROM dependencies AS final

# Copy the rest of the application code
COPY . .

# Create entrypoint script
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]