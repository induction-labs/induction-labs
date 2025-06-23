{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
    # Requires ffmpeg4 because of this https://github.com/gcanat/video_reader-rs/issues/4
    pkgs.ffmpeg_4.dev
    pkgs.llvmPackages_latest.clang # your C compiler
    pkgs.llvmPackages_latest.libclang # for bindgen
    pkgs.pkg-config # to find libavutil.pc
    pkgs.glibc.dev # <-- brings in stdint.h, etc.
    pkgs.zlib.dev # <-- bring in other common headers
  ];
  env.DECORD_REWIND_RETRY_MAX = "64";
  env.BUILDX_BAKE_ENTITLEMENTS_FS = "0";
  # https://discourse.nixos.org/t/libclang-path-and-rust-bindgen-in-nixpkgs-unstable/13264
  env.LIBCLANG_PATH = "${pkgs.llvmPackages_16.libclang.lib}/lib";

  # env.LD_LIBRARY_PATH = lib.makeLibraryPath [pkgs.ffmpeg];

  languages.rust = {
    enable = true;
  };
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
