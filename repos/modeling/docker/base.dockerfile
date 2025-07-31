# syntax=docker/dockerfile:1.7-labs 
# https://docs.docker.com/reference/dockerfile/#copy---parents
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 as base
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

COPY repos/modeling/docker-entrypoint.sh /workspace/repos/modeling/docker-entrypoint.sh
RUN chmod +x /workspace/repos/modeling/docker-entrypoint.sh
ENTRYPOINT [ "/workspace/repos/modeling/docker-entrypoint.sh" ]
# Set the shell to use devenv
SHELL ["devenv", "shell", "--", "/bin/bash", "-c"]

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

# TODO: Use different FROM stages


# COPY ../../ ../../
# For now don't sync vllm :/
# Use --offline to prevent devenv from trying to refresh the environment
