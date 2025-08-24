from setuptools import setup, find_packages

setup(
    name="scrna_longformer",
    version="0.0.0",
    description="Minimal scrna-longformer package for CI/dev",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
