variable "COMMIT_SHA" {
    default = "latest"
}

variable "REGISTRY" {
    default = "us-central1-docker.pkg.dev/induction-labs/induction-docker"
}

target "local" {
    context = "../../"
    dockerfile = "repos/modeling/Dockerfile"
    tags = [
        "modeling:latest",
        "modeling:${COMMIT_SHA}",
    ]
    args = {
    }
    secret = [
    "id=wandb_key,src=./secrets/wandb_key",
    "id=huggingface_key,src=./secrets/huggingface_key",
    "id=gcp_service_account_key,src=./secrets/gcp_service_account_key.json"
    ]
    ssh = ["default"]  # Add this line
}

target "remote" {
    inherits = ["local"]
    tags = [
        "${REGISTRY}/modeling:latest",
        "${REGISTRY}/modeling:${COMMIT_SHA}",
    ]
    push = true
    cache-from = ["type=gha"]
    cache-to = ["type=gha,mode=max"]
}

group "default" {
    targets = ["remote"]
}

# CI-specific target
target "ci" {
    inherits = ["remote"]
    tags = [
        "${REGISTRY}/modeling:${COMMIT_SHA}",
        "${REGISTRY}/modeling:latest",
    ]
}