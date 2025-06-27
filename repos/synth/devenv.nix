{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
    pkgs.zlib
  ];

  languages.python = {
    libraries = [
      # pkgs.cmake
    ];
    enable = true;
    package = pkgs.python312Full;

    uv = {
      enable = true;
    };
    venv.enable = true;
  };
}
