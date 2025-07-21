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

  git-hooks.hooks = {
    shellcheck.enable = true;
    ruff.enable = true;
    ruff-format.enable = true;
    alejandra.enable = true;
  };

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
