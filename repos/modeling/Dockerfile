# syntax=docker/dockerfile:1.7-labs 
# https://docs.docker.com/reference/dockerfile/#copy---parents
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install linux \
  --extra-conf "sandbox = false" \
  --init none \
  --no-confirm

ENV PATH="${PATH}:/nix/var/nix/profiles/default/bin"
RUN nix profile install nixpkgs#devenv


WORKDIR /workspace
# First copy only the stuff needed to setup the environment.
COPY repos/modeling/devenv.* ./repos/modeling/

WORKDIR /workspace/repos/modeling
# Setup the devenv shell. This install all non-cuda deps needed. 
# This will be cached for future RUN commands.

RUN devenv shell
SHELL ["devenv", "shell", "--", "/bin/bash", "-c"]

# Check that ssh is mounted properly
RUN --mount=type=ssh \
  ssh-add -l

# Need to run this otherwise git will fail to clone private repos.
# https://stackoverflow.com/questions/43418188/ssh-agent-forwarding-during-docker-build
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# We need the parents path otherwise it will overwrite at root level.
COPY --parents ./repos/*/pyproject.toml /workspace/
# COPY --parents ./repos/*/README.md /workspace/

COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock
# COPY README.md /workspace/README.md

# We need to set PRETEND_VERSION because VLLM is dynamically versioned
# based on git commit, and we don't copy the .git directory into the container.
# See https://github.com/pypa/setuptools-scm/issues/771
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

ENV UV_LINK_MODE=copy


RUN --mount=type=cache,target=/root/.cache/uv --mount=type=ssh \
  uv sync --locked --no-install-workspace --group flash-attn

# # TODO: Make this build depend on vllm build, so we dont do this giant copy and have a huge layer.
COPY repos/modeling/ /workspace/repos/modeling/
COPY repos/synapse/ /workspace/repos/synapse/


RUN --mount=type=cache,target=/root/.cache/uv --mount=type=ssh \
  uv sync --locked --group flash-attn

# TODO: Don't bake these into the image, mount them as secrets at runtime.
RUN --mount=type=secret,id=wandb_key \
  wandb login $(cat /run/secrets/wandb_key)


RUN --mount=type=secret,id=huggingface_key \
  huggingface-cli login --token $(cat /run/secrets/huggingface_key)

RUN --mount=type=secret,id=gcp_service_account_key \
  gcloud auth activate-service-account --key-file=/run/secrets/gcp_service_account_key

RUN --mount=type=secret,id=gcp_service_account_key \
  gcloud config set project "induction-labs"

RUN mkdir -p /secrets
RUN --mount=type=secret,id=gcp_service_account_key \
  cp /run/secrets/gcp_service_account_key /secrets/gcp-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-service-account.json
# COPY ../../ ../../
# For now don't sync vllm :/

ENTRYPOINT [ "devenv", "shell" ]
# CMD ["pytest"]

CMD uv run python -m src.modeling.train_llm \
  --experiment-name gpt2-alpaca-single-gpu-$(date +%Y-%m-%dT%H-%M-%S) \
  --dataset-name tatsu-lab/alpaca \
  --model-name openai-community/gpt2 \
  --num-epochs 1
