"""Check that packages from requirements.txt are importable and print versions.
Run this after activating your venv to see which packages are missing.
"""
import importlib, sys, pkgutil

mapping = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scanpy": "scanpy",
    "anndata": "anndata",
    "umap-learn": "umap",
    "pyyaml": "yaml",
    "matplotlib": "matplotlib",
}

def check(name, modname):
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, "__version__", None)
        print(f"OK  - {name} (import {modname}) version={ver}")
    except Exception as e:
        print(f"MISS- {name} (import {modname}) -> {e.__class__.__name__}: {e}")

def main():
    print("Checking requirements (import availability and version):\n")
    for line in open("requirements.txt").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("<=")[0].split(">=")[0].split("~=")[0].strip()
        modname = mapping.get(pkg, pkg)
        check(pkg, modname)

if __name__ == "__main__":
    main()
