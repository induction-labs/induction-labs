{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/packages/
  packages = with pkgs; [
    netcat
  ];

  # https://devenv.sh/languages/
  languages.javascript = {
    enable = true;
    pnpm = {
      enable = true;
    };
  };

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    prettier.enable = true;
    eslint.enable = true;
    shellcheck.enable = true;
  };
}
