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
}

target "default" {
    inherits = ["local"]
    tags = [
        "${REGISTRY}/modeling:latest",
        "${REGISTRY}/modeling:${COMMIT_SHA}",
    ]
    push = true
    cache-from = ["type=gha"]
    cache-to = ["type=gha,mode=max"]
}

# CI-specific target
target "ci" {
    inherits = ["remote"]
    tags = [
        "${REGISTRY}/modeling:${COMMIT_SHA}",
        "${REGISTRY}/modeling:latest",
    ]
}