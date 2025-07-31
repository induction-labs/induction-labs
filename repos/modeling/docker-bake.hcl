variable "COMMIT_SHA" {
  default = "latest"
}

variable "REGISTRY" {
  default = "us-central1-docker.pkg.dev/induction-labs/induction-docker"
}
target "base" {
  context    = "../../"
  dockerfile = "repos/modeling/docker/base.dockerfile"
  tags = [
    "modeling-base:latest",
    "modeling-base:${COMMIT_SHA}",
  ]
  args = {}
  ssh  = ["default"] # Add this line
}

target "mdl" {
  contexts = {
    base_image = "target:base"
  }
  context    = "../../"
  dockerfile = "repos/modeling/docker/mdl.dockerfile"
  args       = {}

  tags = ["modeling-mdl:latest"]
  secret = [
    "id=wandb_key,src=./secrets/wandb_key",
    "id=huggingface_key,src=./secrets/huggingface_key",
    "id=gcp_service_account_key,src=./secrets/gcp_service_account_key.json"
  ]
  ssh = ["default"] # Add this line
}

target "eval" {
  contexts = {
    base_image = "target:base"
  }
  context    = "../../"
  dockerfile = "repos/modeling/docker/eval.dockerfile"
  args       = {}
  tags       = ["modeling-eval:latest"]
  secret = [
    "id=wandb_key,src=./secrets/wandb_key",
    "id=huggingface_key,src=./secrets/huggingface_key",
    "id=gcp_service_account_key,src=./secrets/gcp_service_account_key.json"
  ]
  ssh = ["default"] # Add this line
}

group "default" {
  targets = ["mdl"]
}
