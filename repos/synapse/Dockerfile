# syntax=docker/dockerfile:1.7-labs 
# https://docs.docker.com/reference/dockerfile/#copy---parents
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install linux \
  --extra-conf "sandbox = false" \
  --init none \
  --no-confirm

ENV PATH="${PATH}:/nix/var/nix/profiles/default/bin"
RUN nix profile install nixpkgs#devenv


WORKDIR /workspace
# First copy only the stuff needed to setup the environment.
COPY repos/synapse/devenv.* ./repos/synapse/

WORKDIR /workspace/repos/synapse
# Setup the devenv shell. This install all non-cuda deps needed. 
# This will be cached for future RUN commands.
RUN devenv shell
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
# RUN uv sync --locked --no-group workspace --no-install-workspace
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --no-install-workspace

# TODO: Make this build depend on vllm build, so we dont do this giant copy and have a huge layer.
COPY repos/synapse/ /workspace/repos/synapse/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked

# COPY ../../ ../../
# For now don't sync vllm :/

ENTRYPOINT [ "devenv", "shell" ]
# CMD ["pytest"]

CMD uv run python -m synapse.video_loader.main run 

# docker run us-central1-docker.pkg.dev/induction-labs/induction-docker/synapse:latest "uv run python -m synapse.video_loader.main test"