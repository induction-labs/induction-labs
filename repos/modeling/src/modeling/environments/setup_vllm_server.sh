#!/usr/bin/env bash

curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --determinate --no-confirm

# shellcheck disable=SC1091
. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
nix profile install nixpkgs#devenv
git clone --recurse-submodules -j8 git@github.com:induction-labs/induction-labs.git
cd induction-labs/repos/modeling || exit

devenv shell
uv sync --no-install-workspace --group vllm