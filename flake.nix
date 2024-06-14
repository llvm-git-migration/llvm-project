{
  inputs = {
    flake-utils.follows = "nix-vscode-extensions/flake-utils";
    nixpkgs.follows = "nix-vscode-extensions/nixpkgs";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = inputs.nixpkgs.legacyPackages.${system};

          devShells.default = pkgs.mkShell {
            buildInputs = [ 
              packages.default 
              pkgs.git pkgs.ripgrep pkgs.gdb pkgs.clang-tools
              pkgs.cmake pkgs.ninja pkgs.clang pkgs.lld pkgs.chrpath
              pkgs.fish pkgs.neovim pkgs.git pkgs.libclang
              pkgs.ccache
            ] ;
            shellHook = ''
              export CC=clang
              export CXX=clang++
            '';
          };
        in
        {
          inherit packages devShells;
        });
}
