{
  description = "multi-stark Nix flake (Rust)";

  inputs = {
    # System packages, follows fenix so we stay in sync
    nixpkgs.follows = "fenix/nixpkgs";

    # Helper: flake-parts for easier outputs
    flake-parts.url = "github:hercules-ci/flake-parts";

    # Rust toolchain pinned from rust-toolchain.toml
    fenix.url = "github:nix-community/fenix";

    crane.url = "github:ipetkov/crane";
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-parts,
      fenix,
      crane,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        {
          system,
          pkgs,
          ...
        }:
        let
          # Pins the Rust toolchain
          rustToolchain = fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = "sha256-P30Tm3O7vQAE725YtDCDHGjNrSsfZO4us11UwJGZSJo=";
          };

          craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
          src = craneLib.cleanCargoSource ./.;
          craneArgs = {
            inherit src;
            pname = "multi-stark";
            version = "0.1.0";
            strictDeps = true;

            buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];
          };
          cargoArtifacts = craneLib.buildDepsOnly craneArgs;

          multiStark = craneLib.buildPackage (
            craneArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "--locked --features parallel";
            }
          );
        in
        {
          packages.default = multiStark;

          # `nix flake check` runs the test suite (the CI entrypoint).
          # CARGO_PROFILE picks the profile (crane passes it as
          # --cargo-profile); dev-ci matches ci.yml's optimized test build.
          checks.tests = craneLib.cargoNextest (
            craneArgs
            // {
              inherit cargoArtifacts;
              CARGO_PROFILE = "dev-ci";
              cargoNextestExtraArgs = "--workspace --features parallel";
            }
          );

          # Rust shell for host development (`cargo build`, `cargo test`).
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              rustToolchain
              rust-analyzer
              cargo-deny
              cargo-nextest
            ];
          };

          formatter = pkgs.alejandra;
        };
    };
}
