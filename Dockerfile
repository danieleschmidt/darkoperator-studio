# DarkOperator Studio - Production Docker Image
FROM nvidia/cuda:11.8-devel-ubuntu22.04

LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL description="Neural Operators for Ultra-Rare Dark Matter Detection"
LABEL version="0.1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    unzip \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create non-root user
RUN useradd -m -s /bin/bash darkoperator
USER darkoperator
WORKDIR /home/darkoperator

# Copy environment configuration
COPY --chown=darkoperator:darkoperator environment.yml .
COPY --chown=darkoperator:darkoperator requirements.txt .

# Create conda environment
RUN conda env create -f environment.yml \
    && conda clean -afy

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "darkoperator", "/bin/bash", "-c"]

# Copy source code
COPY --chown=darkoperator:darkoperator . /home/darkoperator/darkoperator-studio/
WORKDIR /home/darkoperator/darkoperator-studio

# Install package in development mode
RUN pip install -e .

# Create data directories
RUN mkdir -p data models logs cache results

# Set default configuration
ENV DARKOPERATOR_CACHE_DIR=/home/darkoperator/darkoperator-studio/cache
ENV DARKOPERATOR_MODEL_CACHE=/home/darkoperator/darkoperator-studio/models
ENV DARKOPERATOR_LOG_DIR=/home/darkoperator/darkoperator-studio/logs
ENV DARKOPERATOR_DEVICE=cuda
ENV DARKOPERATOR_MAX_BATCH_SIZE=32

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD conda run -n darkoperator darkoperator --version || exit 1

# Expose ports for Jupyter and API
EXPOSE 8888 8000

# Default command
CMD ["conda", "run", "-n", "darkoperator", "darkoperator", "--help"]

# Multi-stage build for production
FROM darkoperator-base AS production

# Copy only necessary files for production
COPY --from=darkoperator-base --chown=darkoperator:darkoperator \
    /home/darkoperator/darkoperator-studio/darkoperator \
    /home/darkoperator/darkoperator-studio/darkoperator

COPY --from=darkoperator-base --chown=darkoperator:darkoperator \
    /home/darkoperator/darkoperator-studio/setup.py \
    /home/darkoperator/darkoperator-studio/

# Remove development dependencies
RUN conda run -n darkoperator pip uninstall -y pytest pytest-cov black flake8 mypy

# Production entrypoint
ENTRYPOINT ["conda", "run", "-n", "darkoperator", "darkoperator"]