{
  description = "A development environment for the Jolt zkVM Rust project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }@inputs:
    
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;

      in
      {
        devShells.default = pkgs.mkShell {
          
          packages = [
            rustToolchain
            pkgs.rust-analyzer  # For IDE support
            pkgs.cargo-watch    # For auto-reloading builds
          ];

          nativeBuildInputs = [
            pkgs.openssl
            pkgs.pkg-config
            pkgs.zlib 
          ];

        };
      }
    );
}
