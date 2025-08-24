FROM condaforge/mambaforge:latest

WORKDIR /workspace

# Copy repo
COPY . /workspace

# Create and activate environment, install core deps via mamba
RUN mamba create -n scrna -y python=3.10 && \
    /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate scrna && \
      mamba install -y -c conda-forge scanpy anndata umap-learn python-igraph leidenalg pyyaml numpy pandas scikit-learn matplotlib && \
      mamba install -y -c pytorch -c conda-forge pytorch torchvision torchaudio cpuonly || true && \
      python -m pip install --upgrade pip setuptools wheel && \
  python -m pip install -r requirements.txt || true && \
  # install the local package in editable mode so scripts can import scrna_longformer
  python -m pip install -e src || true"

SHELL ["/bin/bash", "-lc"]
ENV PATH=/opt/conda/envs/scrna/bin:$PATH

CMD ["python","scripts/check_requirements.py"]
