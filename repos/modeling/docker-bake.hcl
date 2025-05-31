target "default" {
    context = "../../"  # Can reference parent directories
    dockerfile = "repos/modeling/Dockerfile"
    tags = [
        "modeling:latest",
        "modeling:0.1.0"
    ]
}