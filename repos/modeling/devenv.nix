{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/basics/
  env.GREET = "devenv";
  env.UV = "1";
  # If you turn this off, recompile with uv cache clean first.
  env.VLLM_USE_PRECOMPILED = "1";
  # Set this for cuda kernels we need to compile
  # https://github.com/ssakar/tutorial/blob/main/vllm/Dockerfile
  env.TORCH_CUDA_ARCH_LIST = "12.0+PTX";

  # https://devenv.sh/packages/
  packages = with pkgs; [
  ];

  # https://devenv.sh/languages/
  languages.python = {
    libraries = [
      # pkgs.cmake
    ];
    enable = true;
    uv = {
      enable = true;
    };
    venv.enable = true;
  };
  # languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    hello
    git --version
    # We need to set this manually otherwise triton tries to call `ldconfig` which is UB in nix.
    export TRITON_LIBCUDA_PATH="$LD_LIBRARY_PATH";
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    uv sync --all-extras
    pytest
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
