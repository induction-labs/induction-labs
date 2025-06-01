target "local" {
    context = "../../"
    dockerfile = "repos/modeling/Dockerfile"
    tags = [
        "modeling:latest",
        "modeling:0.1.0",
    ]
    # Builds locally, no push
}

target "remote" {
    inherits = ["local"]
    tags = [
        "us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest",
        "us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:0.1.0",
    ]
    push = true
}

# Group target to build both
group "default" {
    targets = [ "remote"]
}