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
    "induction-base:latest",
    "induction-base:${COMMIT_SHA}",
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

  tags = ["induction-mdl:latest"]
  secret = [
    "id=wandb_key,src=./secrets/wandb_key",
    "id=huggingface_key,src=./secrets/huggingface_key",
    "id=gcp_service_account_key,src=./secrets/gcp_service_account_key.json"
  ]
  ssh = ["default"] # Add this line
}

target "eve" {
  contexts = {
    base_image = "target:base"
  }
  context    = "../../"
  dockerfile = "repos/modeling/docker/eve.dockerfile"
  args       = {}
  tags       = ["induction-eve:latest"]
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
