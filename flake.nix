{
  description = "Brush: 3D Reconstruction engine using Gaussian Splatting";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust Toolchain definieren (Brush benötigt Rust 1.88+)
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Notwendige Bibliotheken für wgpu / Grafik
        runtimeDeps = with pkgs; [
          vulkan-loader
          libxkbcommon
          wayland
          libx11
          libxcursor
          libxi
          libxrandr
        ];

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          makeWrapper
          cmake # Falls C-Abhängigkeiten vorhanden sind
        ];

        buildInputs = runtimeDeps;

      in
      {
        # Entwicklungsumgebung
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          # LD_LIBRARY_PATH ist für wgpu/Vulkan unter Linux oft notwendig
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeDeps}:$LD_LIBRARY_PATH
          '';
        };

        # Paket-Definition (nix build)
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "brush";
          version = "0.3.0"; # Version anpassen

          src = ./.;

          # cargoLock muss vorhanden sein. Falls nicht, 'cargoHash' nutzen.
          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit nativeBuildInputs buildInputs;

          # Nach dem Bauen die Library-Pfade für die Binary setzen
          postInstall = ''
            wrapProgram $out/bin/brush \
              --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath runtimeDeps}
          '';

          meta = with pkgs.lib; {
            description = "3D Reconstruction engine using Gaussian splatting";
            homepage = "https://github.com/ArthurBrussee/brush";
            license = licenses.asl20;
          };
        };
      }
    );
}
