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

# We need the parents path otherwise it will overwrite at root level.
COPY --parents ./repos/*/pyproject.toml /workspace/

COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock

# WORKDIR /workspace/repos/modeling
# We need to set PRETEND_VERSION because VLLM is dynamically versioned
# based on git commit, and we don't copy the .git directory into the container.
# See https://github.com/pypa/setuptools-scm/issues/771
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

# Print amount of disk space available
RUN df -h

RUN uv sync --frozen --no-group workspace --no-install-workspace

# TODO: Make this build depend on vllm build, so we dont do this giant copy and have a huge layer.
COPY repos/modeling/ /workspace/repos/modeling/
# COPY ../../ ../../
# For now don't sync vllm :/
# RUN uv sync --frozen

CMD ["devenv", "shell", "pytest"]
