FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone necessary repositories
RUN git clone https://github.com/facebookresearch/pifuhd.git
WORKDIR /app/pifuhd
RUN sh ./scripts/download_trained_model.sh
WORKDIR /app

RUN git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
WORKDIR /app/lightweight-human-pose-estimation.pytorch
RUN wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
WORKDIR /app

# Fix the torch.load issue in pifuhd
RUN sed -i "s/torch.load(state_dict_path, map_location=cuda)/torch.load(state_dict_path, map_location=cuda, weights_only=False)/g" /app/pifuhd/apps/recon.py

# Set Python path to include both repositories
ENV PYTHONPATH=/app/pifuhd:/app/lightweight-human-pose-estimation.pytorch:$PYTHONPATH

# Copy the FastAPI application
COPY main.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]
