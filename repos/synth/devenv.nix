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
    package = pkgs.python312;
  };
}
