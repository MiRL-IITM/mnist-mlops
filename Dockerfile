FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# If you need specific CUDA libraries like cuDNN, use:
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "-m", "src.api.app"]