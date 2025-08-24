CI checks: host vs container
=============================

This repository's CI intentionally runs two complementary checks:

1) Host-based setup (`setup-check` job)
- Uses Mambaforge on the GitHub runner to create a `scrna` environment and install packages via `mamba`.
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
