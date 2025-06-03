# Installation

## Cloning
Make sure to clone this repository with the `--recurse-submodules` flag to ensure all submodules are initialized correctly:
```bash
git clone --recurse-submodules
# Or if you already cloned it without submodules:
git submodule update --init --recursive
```

## Installation
Everything in this repo is managed through [devenv](https://devenv.sh/getting-started/).
To install devenv first install [nix](https://github.com/DeterminateSystems/nix-installer):
```bash
curl -fsSL https://install.determinate.systems/nix | sh -s -- install --determinate
```
Then install devenv:
```bash
nix profile install nixpkgs#devenv
```




# Repository Structure. 
We use [Hatchling](https://hatch.pypa.io/latest/) to build python packages, with toplevel uv project configuration. This is why individual repo code needs to be nested like `action-space/repos/<repo_name>/src/<repo_name>/src_code.py`. It is unfortunate that we need to nest like this but that is how hatchling works.