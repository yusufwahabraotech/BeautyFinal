# Docker Build & Deployment Guide for BeautyFinal

## Project Summary
**BeautyFinal** is a FastAPI-based REST API that measures human body dimensions from photos:
- **Inputs:** Front and side view images
- **Processing:** Uses PIFuHD (3D body reconstruction) + pose estimation
- **Outputs:** 7 measurements (shoulder width, waist/bust/hip/neck circumference, arm length, inseam)

## Prerequisites
- Docker installed (`docker --version`)
- At least 20GB free disk space (for models)
- GPU access recommended (CUDA 12.1)

## Building the Docker Image

### 1. Build locally
```bash
cd /workspaces/BeautyFinal
docker build -t beauty-api:latest .
```

**Build time:** ~15-20 minutes (downloads ~6GB of models)

### 2. Test locally
```bash
docker run --gpus all -p 8000:8000 beauty-api:latest
```

Visit `http://localhost:8000` - should see API status

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Process images (requires front.jpg and side.jpg)
curl -X POST "http://localhost:8000/process" \
  -F "front_image=@front.jpg" \
  -F "side_image=@side.jpg"
```

## Deployment Options

### Option A: Google Cloud Run (Recommended)
**Pros:** Free tier, auto-scaling, serverless
**Cons:** Memory/timeout limits (max 4GB, 60min timeout)

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# 3. Tag and push image
docker tag beauty-api:latest gcr.io/YOUR_PROJECT_ID/beauty-api:latest
docker push gcr.io/YOUR_PROJECT_ID/beauty-api:latest

# 4. Deploy to Cloud Run
gcloud run deploy beauty-api \
  --image gcr.io/YOUR_PROJECT_ID/beauty-api:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 3600 \
  --allow-unauthenticated
```

### Option B: Google Compute Engine (Better for GPU)
**Pros:** Full control, GPU support, better performance
**Cons:** Cost (but free tier available)

```bash
# 1. Create VM instance
gcloud compute instances create beauty-api \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# 2. SSH into instance
gcloud compute ssh beauty-api --zone=us-central1-a

# 3. Inside VM, clone and build:
git clone <your-repo>
cd BeautyFinal
docker build -t beauty-api .
docker run --gpus all -p 8000:8000 beauty-api

# 4. Access via External IP
gcloud compute instances describe beauty-api --zone=us-central1-a | grep natIP
```

### Option C: AWS ECS/Fargate
```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker tag beauty-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/beauty-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/beauty-api:latest

# 2. Deploy via ECS/Fargate with web UI
```

### Option D: Docker Hub + Self-hosted Server
```bash
# 1. Push to Docker Hub
docker tag beauty-api:latest YOUR_USERNAME/beauty-api:latest
docker login
docker push YOUR_USERNAME/beauty-api:latest

# 2. On server:
docker run -d --gpus all -p 8000:8000 YOUR_USERNAME/beauty-api:latest
```

## Production Checklist

- [ ] Add authentication (remove `allow-unauthenticated` or add API key)
- [ ] Set up HTTPS/SSL certificate
- [ ] Add rate limiting
- [ ] Set up monitoring/logging (Google Cloud Logging, Datadog, etc.)
- [ ] Configure backup storage for results
- [ ] Set up health checks and auto-restart
- [ ] Test with real images from target use case

## API Documentation

**Base URL:** `http://YOUR_DEPLOYED_URL`

### Endpoints

#### GET /
Returns API status

#### GET /health
Returns health status and GPU availability

#### POST /process
**Body:** Multipart form data
- `front_image` (file): JPEG/PNG of front view
- `side_image` (file): JPEG/PNG of side view

**Response:**
```json
{
  "success": true,
  "measurements": {
    "shoulder_width_cm": 42.5,
    "shoulder_width_inches": 16.73,
    "waist_circumference_cm": 81.2,
    "waist_circumference_inches": 31.97,
    "hip_circumference_cm": 95.3,
    "hip_circumference_inches": 37.52,
    "bust_circumference_cm": 91.8,
    "bust_circumference_inches": 36.14,
    "neck_circumference_cm": 38.5,
    "neck_circumference_inches": 15.16,
    "arm_length_cm": 59.2,
    "arm_length_inches": 23.31,
    "inseam_cm": 78.5,
    "inseam_inches": 30.91
  },
  "message": "Body measurements calculated successfully"
}
```

## Troubleshooting

**Image:** `docker images | grep beauty`

**Logs:** `docker logs <container_id>`

**GPU not detected:** Check Docker daemon settings
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu20.04 nvidia-smi
```

**Out of memory:** Reduce batch size in code or increase container memory limit

**Model download fails:** Check internet connection; models total ~6GB
