# syntax=docker/dockerfile:1.7-labs 
# ARG BASE_IMAGE
# FROM ${BASE_IMAGE}
FROM base_image as eval
# Here we first sync *without* workspace and ssh required deps, because those always need docker rebuild.
# NOTE: This stage does not require ssh mount
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-workspace --group evals --no-group ssh-required
# New we add secrets

RUN mkdir -p /secrets
# TODO: Don't bake these into the image, mount them as secrets at runtime.
RUN --mount=type=secret,id=wandb_key \
  wandb login $(cat /run/secrets/wandb_key)

RUN --mount=type=secret,id=huggingface_key \
  huggingface-cli login --token $(cat /run/secrets/huggingface_key)

RUN --mount=type=secret,id=eval_gcp_secret_key \
  gcloud auth activate-service-account --key-file=/run/secrets/eval_gcp_secret_key

RUN --mount=type=secret,id=eval_gcp_secret_key \
  gcloud config set project "induction-labs"

RUN --mount=type=secret,id=eval_gcp_secret_key \
  cp /run/secrets/eval_gcp_secret_key /secrets/gcp-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-service-account.json


# # TODO: Make this build depend on vllm build, so we dont do this giant copy and have a huge layer.
COPY repos/modeling/ /workspace/repos/modeling/
COPY repos/synapse/ /workspace/repos/synapse/

# Check that ssh is mounted properly
RUN --mount=type=ssh \
  ssh-add -l
# Need to run this otherwise git will fail to clone private repos.
# https://stackoverflow.com/questions/43418188/ssh-agent-forwarding-during-docker-build
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Now we sync *with* ssh required deps, which will always require a rebuild.
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=ssh \
  uv sync --frozen --group evals --group ssh-required

ENTRYPOINT [ "/docker-entrypoint.sh" ]

CMD ["eve", "--help"]