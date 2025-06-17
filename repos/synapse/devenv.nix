{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [];
  env.DECORD_REWIND_RETRY_MAX = "64";

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
}
