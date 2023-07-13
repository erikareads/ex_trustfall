{ pkgs ? import <nixpkgs> {} }:

    with pkgs;
    let
      elixir = beam.packages.erlangR25.elixir_1_14;
      in
      mkShell {
          buildInputs = [ elixir beamPackages.rebar3 
    rustc
    cargo
    rustfmt
    rust-analyzer
    clippy
    pkg-config
    alsa-lib

];
  RUST_BACKTRACE = 1;
      }
