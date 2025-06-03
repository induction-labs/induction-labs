# Docker build:
`docker buildx bake`

# Configure Google Artifact Registry:
`gcloud auth configure-docker us-central1-docker.pkg.dev`

# Run with GPUs:
`docker run --device nvidia.com/gpu=all docker.io/library/modeling:latest`

# Push to Google Artifact Registry:
`docker push us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest`



# Secrets:
Need secrets in `induction-labs/repos/modeling/secrets/...`.
## WanDB:
Add wandb API key to `induction-labs/repos/modeling/secrets/wandb_key`
Find it in `cat ~/.netrc`, you should see
```sh
machine wandb.ai
  login <your_wandb_username>
  password <your_wandb_api_key>
```
put `<your_wandb_api_key>` in `induction-labs/repos/modeling/secrets/wandb_key` as plain text.