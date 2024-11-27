{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  packages = [
    just
    cmake
    gnumake
    gcc
    clang-tools
    gdb
    cling
  ];
}
