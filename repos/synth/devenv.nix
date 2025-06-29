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
    ];
    enable = true;
    uv = {
      enable = true;
    };
    venv.enable = true;
  };
}
