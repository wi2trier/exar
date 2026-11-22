{
  lib,
  stdenv,
  callPackage,
  fetchFromGitHub,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python314,
  ty,
  basedpyright,
  onetbb,
  cacert,
  libstemmer,
  zlib,
}:
let
  pdocRepo = fetchFromGitHub {
    owner = "mitmproxy";
    repo = "pdoc";
    tag = "v16.0.0";
    hash = "sha256-9amp6CWYIcniVfdlmPKYuRFR7B5JJtuMlOoDxpfvvJA=";
  };
  uvLock = lib.importTOML ./uv.lock;
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ./.;
    uvLock = uvLock // {
      # uv2nix rejects default-groups, and Nix environments select groups explicitly.
      package = map (package: removeAttrs package [ "default-groups" ]) uvLock.package;
    };
  };
  projectOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  getCudaPkgs = attrs: lib.filter (name: lib.hasPrefix "nvidia-" name) (lib.attrNames attrs);
  cudaOverlay =
    final: prev:
    lib.genAttrs (getCudaPkgs prev) (
      name:
      prev.${name}.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps = true;
      })
    );
  packageOverlay =
    final: prev:
    lib.mapAttrs (name: value: prev.${name}.overrideAttrs value) {
      torch = old: {
        autoPatchelfIgnoreMissingDeps = true;
      };
      numba = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ onetbb ];
      };
      zlib-state = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ zlib ];
      };
      pystemmer = old: {
        # https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/development/python-modules/pystemmer/default.nix
        PYSTEMMER_SYSTEM_LIBSTEMMER = "${lib.getDev libstemmer}/include";
        NIX_CFLAGS_COMPILE = [ "-I${lib.getDev libstemmer}/include" ];
        NIX_CFLAGS_LINK = [ "-L${libstemmer}/lib" ];
      };
      exar = old: {
        meta = (old.meta or { }) // {
          maintainers = with lib.maintainers; [ mirkolenz ];
          license = lib.licenses.mit;
          homepage = "https://github.com/wi2trier/exar";
          description = "Experience-grounded agentic reasoning for any agent runtime";
          platforms = with lib.platforms; darwin ++ linux;
        };
        passthru = lib.recursiveUpdate (old.passthru or { }) {
          tests.pytest = stdenv.mkDerivation {
            name = "${final.exar.name}-pytest";
            inherit (final.exar) src;
            nativeBuildInputs = [
              cacert
              (mkVenv "exar-test-env" {
                exar = [
                  "all"
                  "test"
                ];
              })
            ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              export HOME=$(mktemp -d)
              export NUMBA_CACHE_DIR=$HOME/.numba_cache
              pytest --cov-report=html
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mv htmlcov $out
              runHook postInstall
            '';
          };
          tests.typecheck = stdenv.mkDerivation {
            name = "${final.exar.name}-typecheck";
            inherit (final.exar) src;
            nativeBuildInputs = [
              ty
              basedpyright
              typecheckVenv
            ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              export HOME=$(mktemp -d)
              ty check --python ${typecheckVenv} src
              basedpyright --level error --pythonpath ${typecheckVenv}/bin/python src
              runHook postBuild
            '';
            installPhase = ''
              touch $out
            '';
          };
          docs = stdenv.mkDerivation {
            name = "${final.exar.name}-docs";
            inherit (final.exar) src;
            nativeBuildInputs = [
              cacert
              (mkVenv "exar-docs-env" {
                exar = [
                  "all"
                  "docs"
                ];
              })
            ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild

              python docs/llms.py > llms.txt

              pdoc \
                -d google \
                -t ${pdocRepo}/examples/dark-mode \
                --math \
                -o "$out" \
                ./src/exar

              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall

              cp llms.txt "$out"

              runHook postInstall
            '';
          };
        };
      };
    };
  baseSet = callPackage pyproject-nix.build.packages {
    python = python314;
  };
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      projectOverlay
      cudaOverlay
      packageOverlay
    ]
  );
  mkVenv =
    name: deps:
    (pythonSet.mkVirtualEnv name deps).overrideAttrs (_: {
      venvIgnoreCollisions = [ "${pythonSet.python.sitePackages}/griffe/*" ];
    });
  typecheckVenv = mkVenv "exar-typecheck-env" { exar = [ "all" ]; };
  inherit (callPackage pyproject-nix.build.util { }) mkApplication;
in
mkApplication {
  venv = mkVenv "exar-env" { exar = [ "all" ]; };
  package = pythonSet.exar;
}
