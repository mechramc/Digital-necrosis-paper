# Digital Necrosis - Reproducibility Container
# Primary target: Ubuntu 24.04 LTS with CUDA 12.8+ on RTX 5090

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN python3.12 -m pip install --no-cache-dir -e ".[gpu,analysis]"

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Default entrypoint
ENTRYPOINT ["python3.12"]
CMD ["scripts/run_experiment.py", "--config", "configs/default.yaml", "--seed", "42"]
