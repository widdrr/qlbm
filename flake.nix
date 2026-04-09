{
  description = "QLBM — Quantum Lattice Boltzmann Method";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        tex = pkgs.texlive.combine {
          inherit (pkgs.texlive)
            scheme-small
            # math
            amsmath
            amsfonts
            # physics notation
            braket
            qcircuit
            # page layout
            geometry
            fancyhdr
            parskip
            setspace
            # fonts
            psnfss       # .sty files for PS fonts
            courier      # pcrr8t metrics
            helvetic     # phvr8t metrics
            hyphen-english
            # tables
            booktabs
            multirow
            tools        # longtable, multicol
            # figures
            graphics     # graphicx
            float
            subfig
            pgf          # tikz
            quantikz     # quantum circuits
            xargs        # quantikz dep
            tikz-cd      # quantikz dep
            xstring      # quantikz dep
            environ      # quantikz dep
            # cross-references & links
            hyperref
            # code listings
            listings
            # nomenclature & indexing
            nomencl
            bigfoot      # perpage
            # build
            latexmk;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "qlbm";
          packages = with pkgs; [
            uv
            tex
            poppler-utils
          ];

        };
      }
    );
}
