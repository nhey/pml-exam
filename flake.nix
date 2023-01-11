{
  description = "Axons";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
  };

  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
    };

    pyEnv = pkgs.python3.withPackages (ps: with ps; [
      notebook
      jupyterlab
      ipywidgets
      numpy
      matplotlib
      scikit-learn
      statsmodels
      pytorch
      torchvision
      scipy
      pyro-ppl
      arviz
    ]);

    jupyterWrap = pkgs.writeShellScriptBin "jupyterWrap" ''
      ${pkgs.xterm}/bin/xterm -fg white -bg black -e '${pyEnv}/bin/jupyter notebook' &
    '';
  in {
    defaultPackage.x86_64-linux = jupyterWrap;

    devShell.x86_64-linux = with pkgs; mkShell {
      buildInputs = [
        pyEnv
      ];
      shellHook = ''
        export PS1="''${PS1}Nix> "
      '';
    };

  };
}
