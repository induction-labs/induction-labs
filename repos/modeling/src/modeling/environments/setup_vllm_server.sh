#!/usr/bin/env bash

# curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --determinate --no-confirm

# shellcheck disable=SC1091
# . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
# nix profile install nixpkgs#devenv
git clone --recurse-submodules -j8 git@github.com:induction-labs/induction-labs.git
cd induction-labs/repos/modeling || exit

# # use devenv cache
# LINE1='extra-substituters = https://devenv.cachix.org'
# LINE2='extra-trusted-public-keys = devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw='

# # Append to file using sudo
# echo "$LINE1" | sudo tee -a /etc/nix/nix.custom.conf > /dev/null
# echo "$LINE2" | sudo tee -a /etc/nix/nix.custom.conf > /dev/null

# devenv shell

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-install-workspace --group evals

STEP=50
CKPT_PATH="induction-labs/checkpoints/uitars_sft_7b/2025-07-29T15-03-50.s34UyE2B"
mkdir step_$STEP
gcloud storage cp -r gs://$CKPT_PATH/step_$STEP/ .
# add files into config path (TODO: save these at checkpoint save time)
cp -r src/modeling/environments/uitars_config/* step_$STEP/

# then run ./setup_vllm_multi_tmux.sh (and set the model path to be step_$STEP)
# then run osworld.py (manually set the paths, eval configs, etc)
# the osworld good config path is gs://induction-labs/jonathan/osworld/osworld_subset_solved_by_annotators.json (run gsutil cp gs://induction-labs/jonathan/osworld/osworld_subset_solved_by_annotators.json .)
# then copy from the output path manually to gcs after the eval is done :(