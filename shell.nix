# Used for remote development on a GPU-enabled machine.
#
# uv is the standard way to install just-focus (see pyproject.toml); this shell is
# only for testing on remote hardware running NixOS with an NVIDIA GPU.
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

      zernipax = final.buildPythonPackage {
        pname = "zernipax";
        version = "0.2.1";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/43/95/95798cfe41979ee71ce161734ef4a2a4894461b484332fd75d570076507d/zernipax-0.2.1-py3-none-any.whl";
          hash = "sha256-zkQURQT0KHfDh9rKTLlow3T346SZun9TbJEqOfdKYyA=";
        };

        propagatedBuildInputs = with final; [
          jax
          matplotlib
          mpmath
          numpy
        ];

        pythonImportsCheck = [ "zernipax" ];

        meta = with pkgs.lib; {
          description = "Fast and accurate Zernike polynomial calculator using JAX";
          homepage = "https://github.com/PlasmaControl/ZERNIPAX";
          license = licenses.mit;
        };
      };
    };
  };

  pythonEnv = myPython.withPackages (ps: with ps; [
    matplotlib
    numpy
    pytest
    just-focus
    jax
    zernipax
  ]);
in

pkgs.mkShellNoCC {
  packages = [ pythonEnv ];

  shellHook = ''
    export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  '';
}
