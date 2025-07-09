# Multi-stage build for XAUUSD scalping system
# Optimized for CUDA-ready inference with minimal runtime footprint

# Base stage with CUDA support
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libhdf5-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NUMBA_ENABLE_CUDASIM=0

# Development stage
FROM base AS development

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
COPY docker/requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipdb \
    pytest-xdist \
    pytest-benchmark

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Production stage
FROM base AS production

WORKDIR /app

# Copy only production requirements
COPY requirements.txt .
COPY docker/requirements-docker.txt .

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip cache purge

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .

# Install package
RUN pip install . && rm -rf /root/.cache/pip

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader
RUN chown -R trader:trader /app
USER trader

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models

# Copy entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER trader

ENTRYPOINT ["/entrypoint.sh"]

# Training stage - for model training workloads
FROM production AS trainer

USER root
# Install additional training dependencies
RUN pip install \
    optuna \
    plotly \
    shap \
    seaborn \
    scikit-optimize

# Switch back to trader user
USER trader

# Default command for training
CMD ["python", "scripts/model_trainer.py"]

# Backtesting stage
FROM production AS backtester

# Default command for backtesting
CMD ["python", "scripts/backtest_runner.py"]

# Live trading stage - minimal footprint
FROM production AS live

# Remove unnecessary packages to minimize attack surface
USER root
RUN pip uninstall -y \
    matplotlib \
    seaborn \
    plotly \
    jupyter \
    notebook

USER trader

# Default command for live trading
CMD ["python", "scripts/live_trader.py"]