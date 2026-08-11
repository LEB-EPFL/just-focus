# Used for remote development on a GPU-enabled machine.
let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/archive/fcb8fcd6bf2d0adecae5bd491afaaaf8311b758d.tar.gz";
  pkgs = import nixpkgs { config = { cudaSupport = true; allowUnfree = true; }; overlays = []; };

  pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);

  myPython = pkgs.python314.override {
    self = myPython;
    packageOverrides = final: prev: {
      just-focus = final.mkPythonEditablePackage {
        pname = pyproject.project.name;
        inherit (pyproject.project) version;
        
        root = "$REPO_ROOT/src";

        inherit (pyproject.project) scripts;
      };
    };
  };

  pythonEnv = myPython.withPackages (ps: with ps; [
    matplotlib
    numpy
    pytest
    just-focus
  ]);
in

pkgs.mkShellNoCC {
  packages = [ pythonEnv ];

  shellHook = ''
    export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  '';
}
