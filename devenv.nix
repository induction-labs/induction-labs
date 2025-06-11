{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  env.UV = "1";

  packages = with pkgs; [
  ];

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
