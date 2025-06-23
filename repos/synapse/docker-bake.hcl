variable "COMMIT_SHA" {
    default = "latest"
}

variable "REGISTRY" {
    default = "us-central1-docker.pkg.dev/induction-labs/induction-docker"
}

target "local" {
    context = "../../"
    dockerfile = "repos/synapse/Dockerfile"
    tags = [
        "synapse:latest",
        "synapse:${COMMIT_SHA}",
    ]
}

target "remote" {
    inherits = ["local"]
    tags = [
        "${REGISTRY}/synapse:latest",
        "${REGISTRY}/synapse:${COMMIT_SHA}",
    ]
    push = true
    cache-from = ["type=gha"]
    cache-to = ["type=gha,mode=max"]
}

group "default" {
    targets = ["local", "remote"]
}

# CI-specific target
target "ci" {
    inherits = ["remote"]
    tags = [
        "${REGISTRY}/synapse:${COMMIT_SHA}",
        "${REGISTRY}/synapse:latest",
    ]
}