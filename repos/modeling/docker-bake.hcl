variable "SHORT_SHA" {
  default = "latest"
}

variable "PR_NUMBER" {
  default = ""
}

target "base" {
  context = "../../"
  dockerfile = "repos/modeling/Dockerfile"
}

target "local" {
  inherits = ["base"]
  tags = [
    "modeling:latest",
    "modeling:${SHORT_SHA}",
  ]
}

target "default" {
  inherits = ["base"]
  tags = [
    "us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:${SHORT_SHA}",
    "us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:latest",
    notequal("", PR_NUMBER) ? "us-central1-docker.pkg.dev/induction-labs/induction-docker/modeling:pr-${PR_NUMBER}" : "",
  ]
  push = true
}