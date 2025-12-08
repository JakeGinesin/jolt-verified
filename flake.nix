{
  description = "A development environment for the Jolt zkVM Rust project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    verus-flake.url = "github:JakeGinesin/verus-flake";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, verus-flake, ... }@inputs:
    
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
            pkgs.rust-analyzer  
            pkgs.cargo-watch    
            verus-flake.packages.${system}.default
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
