{ pkgs, lib, config, inputs, ... }:

{
  packages = [ pkgs.git ];

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
