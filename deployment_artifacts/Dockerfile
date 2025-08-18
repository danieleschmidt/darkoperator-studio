FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY darkoperator/ ./darkoperator/
COPY setup.py pyproject.toml ./

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash darkoperator
RUN chown -R darkoperator:darkoperator /app
USER darkoperator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import darkoperator; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "darkoperator.cli", "autonomous"]