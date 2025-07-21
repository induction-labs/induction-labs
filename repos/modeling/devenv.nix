{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/basics/
  # env.GREET = "devenv";
  env.UV = "1";
  # If you turn this off, recompile with uv cache clean first.
  env.VLLM_USE_PRECOMPILED = "1";
  # Set this for cuda kernels we need to compile
  # https://github.com/ssakar/tutorial/blob/main/vllm/Dockerfile
  env.TORCH_CUDA_ARCH_LIST = "12.0+PTX";

  # Allow buildx bake by default to access ../../ context
  env.BUILDX_BAKE_ENTITLEMENTS_FS = "0";
  # env.LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1";

  # https://devenv.sh/packages/
  packages = with pkgs;
    [
      # Keep these here so it is easier to debug the docker image.
    ]
    ++ lib.optionals pkgs.stdenv.isLinux [
      kmod # for lsmod
      strace
      vim
      ffmpeg-full
      (
        pkgs.google-cloud-sdk.withExtraComponents [
          pkgs.google-cloud-sdk.components.gke-gcloud-auth-plugin
        ]
      )

      # TODO: Build depot for all platforms (mac)
      # TODO: Put depot in its own nix flake
      # For now just curl -L https://depot.dev/install-cli.sh | sh
      (let
        version = "2.95.0";
        pname = "depot";
      in
        pkgs.stdenv.mkDerivation rec {
          inherit pname version;

          src = fetchurl {
            url = "https://github.com/depot/cli/releases/download/v${version}/depot_${version}_linux_amd64.tar.gz";
            # You can prefetch to get the right hash, then paste it here:
            sha256 = "sha256-LuFqYtduqFpvaV9xQlKfKPrCuyv0LiMP+9WHIZ8pQQQ=";
          };

          dontBuild = true;
          unpackPhase = ''
            tar xzf $src
            mv bin source
          '';
          # Convert the install script logic here
          installPhase = ''
            mkdir -p $out/bin
            install -m755 source/depot $out/bin/depot
          '';
        })
    ];

  # https://devenv.sh/languages/
  languages.python = {
    libraries = [
    ];
    enable = true;
    uv = {
      enable = true;
    };
    venv.enable = true;
  };
  # languages.rust.enable = true;

  enterShell = ''
    # We need to set this manually otherwise triton tries to call `ldconfig` which is UB in nix.
    export TRITON_LIBCUDA_PATH="$LD_LIBRARY_PATH";
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    uv sync --all-extras
    pytest
  '';
}
