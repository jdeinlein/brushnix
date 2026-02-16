{
  description = "Brush: 3D Reconstruction engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Rust Toolchain
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # --- Dependencies ---
        
        # Common dependncies
        commonNativeInputs = with pkgs; [
          rustToolchain
          pkg-config
          cmake
        ];

        # Linux
        linuxDeps = with pkgs; [
          vulkan-loader
          libxkbcommon
          wayland
          libx11
          libxcursor
          libxi
          libxrandr
        ];

        # macOS
        darwinDeps = [
          pkgs.apple-sdk
          pkgs.libiconv
        ];

        nativeBuildInputs = commonNativeInputs ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.makeWrapper ];
        buildInputs = if pkgs.stdenv.isDarwin then darwinDeps else linuxDeps;

      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          shellHook = ''
            ${if pkgs.stdenv.isDarwin then ''
              # macOS: SDK paths
              export SDKROOT=$(xcrun --show-sdk-path)
            '' else ''
              # Linux: Library paths
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath linuxDeps}:$LD_LIBRARY_PATH
            ''}
            echo "Brush Dev Shell für ${system} geladen."
          '';
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "brush";
          version = "0.3.0";
          src = ./.;

          cargoLock = { lockFile = ./Cargo.lock; };

          inherit nativeBuildInputs buildInputs;

          preConfigure = pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
            export SDKROOT=$(xcrun --show-sdk-path)
          '';

          postInstall = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            wrapProgram $out/bin/brush \
              --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath linuxDeps}
          '';

          doCheck = false;

          meta = with pkgs.lib; {
            description = "3D Reconstruction engine using Gaussian splatting";
            homepage = "https://github.com/ArthurBrussee/brush";
            platforms = platforms.linux ++ platforms.darwin;
          };
        };
      }
    );
}