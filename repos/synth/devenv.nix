{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
  ];

  languages.python = {
    libraries = [
      # Otherwise pandas yells at us for not being able to find libz.so.1
      pkgs.zlib
    ];
    enable = true;
    uv = {
      enable = true;
    };
    venv.enable = true;
  };
}
