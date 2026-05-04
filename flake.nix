{
  description = "multi-stark Nix flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix.url = "github:nix-community/fenix";
    crane.url = "github:ipetkov/crane";
  };

  outputs = inputs @ {
    flake-parts,
    fenix,
    crane,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem = {
        system,
        pkgs,
        ...
      }: let
        rustToolchain = fenix.packages.${system}.fromToolchainFile {
          file = ./rust-toolchain.toml;
          sha256 = "sha256-sqSWJDUxc+zaz1nBWMAJKTAGBuGWP25GCftIOlCEAtA=";
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        src = craneLib.cleanCargoSource ./.;

        commonArgs = {
          inherit src;
          strictDeps = true;
          cargoExtraArgs = "--locked";

          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        multi-stark = craneLib.buildPackage (commonArgs // {inherit cargoArtifacts;});
        multi-stark-tests = craneLib.cargoTest (commonArgs // {inherit cargoArtifacts;});
      in {
        packages = {
          default = multi-stark;
          tests = multi-stark-tests;
        };

        checks = {
          inherit multi-stark multi-stark-tests;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            clang
            rustToolchain
            rust-analyzer
          ];
        };
      };
    };
}
