CI checks: host vs container
=============================

This repository's CI intentionally runs two complementary checks:

1) Host-based setup (`setup-check` job)
- Uses Mambaforge on the GitHub runner to create a `scrna_fixed` environment and install packages via `mamba`.
- Runs `scripts/check_requirements.py` and a small model smoke test.
- This job mirrors the project’s standard preparation script (`preparation/INSTALL.sh`) and is useful for
  checking package availability on the runner environment.

2) Container-based setup (`container-check` job)
- Builds `preparation/Dockerfile` and runs `scripts/check_requirements.py` inside the container image.
- Ensures the Dockerfile is valid and the containerized environment can reproduce the preparation stage.

Why both?
- Local development on macOS often uses hardware-specific builds (e.g., PyTorch MPS). Such builds cannot
  be reproduced in Linux containers. Running both checks captures two important guarantees:
  - host env: developer-friendly, may include platform optimizations (MPS/GPU).
  - container env: reproducible, portable, and suitable for CI/demos across contributors and CI machines.

Practical notes for contributors
- Expect the container check to differ from local behavior when using MPS or CUDA.
- If your local run works (MPS) but container check fails, check whether the failure is due to platform-specific
  binary differences rather than code bugs.

If you want a single-source-of-truth environment for dev and CI, consider using a devcontainer for development
and optionally matching CI to the devcontainer image (note: devcontainer on macOS still runs Linux images).

## Known issue: `docker run scrna-longformer:ci` image not found / access denied

Observed error when CI tried to run:

```
Unable to find image 'scrna-longformer:ci' locally
docker: Error response from daemon: pull access denied for scrna-longformer, repository does not exist or may require 'docker login': denied: requested access to the resource is denied

Error: Process completed with exit code 125.
```

What this means:
- The CI attempted to run a local image tag `scrna-longformer:ci` that wasn't present on the runner.
- The runner doesn't have a local image with that tag, and `docker run` then attempted to pull the image from a remote
  registry (Docker Hub) where the image doesn't exist or is private, producing the access denied error.

How to reproduce locally:
- Build the image locally and try the same command:

```bash
docker build -t scrna-longformer:ci -f preparation/Dockerfile .
docker run --rm scrna-longformer:ci python scripts/check_requirements.py
```

Minimal remediation (deferred):
- Build the image locally on the runner (or in CI) before attempting `docker run`, or push the built image to a registry
  (for example GitHub Container Registry / GHCR) and update CI to pull that tag.
- This repository's `container-check` workflow currently builds the image; if you see the error in a different job, ensure
  the build step ran successfully and the image tag matches the `docker run` invocation.

Notes:
- This is a reproducible, low-priority infrastructure issue (image tag availability) and not a model/code bug. You said
  you'd prefer to focus on model implementation next — this note documents the problem so you can return to it later.
