# Docker build:
`docker buildx bake`

# Configure Google Artifact Registry:
`gcloud auth configure-docker us-central1-docker.pkg.dev`

# Run with GPUs:
`docker run --device nvidia.com/gpu=all docker.io/library/modeling:latest`

# Push to Google Artifact Registry:
`docker push us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest`