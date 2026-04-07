FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

# Copy service code
COPY . .

# Redirect model download caches to the persistent volume
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

CMD ["python3", "main.py"]
