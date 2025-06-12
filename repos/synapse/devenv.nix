{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [];

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
