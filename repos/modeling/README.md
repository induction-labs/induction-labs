# Docker build:
`docker buildx bake`

# Configure Google Artifact Registry:
`gcloud auth configure-docker us-central1-docker.pkg.dev`

# Run with GPUs:

To run on a GPU, first find where `libcuda.so` is located on your system. You can find it with:
```bash
docker run --rm --device nvidia.com/gpu=all us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest find /usr -name "libcuda.so*" 2>/dev/null
```

For example, it might be located at `/lib/x86_64-linux-gnu/libcuda.so` or `/usr/local/nvidia/lib64/libcuda.so`.
Also need to find `libnvidia-ml.so.1`:
```bash
find /usr -name "libnvidia-ml.so.1" 2>/dev/null
```


Then run
```sh
docker run --device nvidia.com/gpu=all -e LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest pytest
```


```sh
docker run --gpus all -it -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest /bin/bash
```


# Push to Google Artifact Registry:
`docker push us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest`

```sh
sudo usermod -aG docker $USER
newgrp docker
gcloud auth configure-docker us-central1-docker.pkg.dev
docker pull us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest
```


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