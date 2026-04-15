FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# --extra-index-url points at PyTorch's CU121 wheels (bundled CUDA libs).
# This avoids pip exploring hundreds of nvidia-* subdep combinations from PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --timeout 300 --retries 10 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# Copy service code
COPY . .

# Redirect model download caches to the persistent volume
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

CMD ["python3", "main.py"]
