{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
    # Requires ffmpeg4 because of this https://github.com/gcanat/video_reader-rs/issues/4
    pkgs.ffmpeg.dev
  ];
  env.BUILDX_BAKE_ENTITLEMENTS_FS = "0";
  # https://discourse.nixos.org/t/libclang-path-and-rust-bindgen-in-nixpkgs-unstable/13264

  # languages.rust = {
  #   enable = true;
  # };
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
