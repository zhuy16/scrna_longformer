#!/bin/bash
# setup_environment.sh - Ensures working environment is activated and verified

set -e  # Exit on any error

echo "🚀 Setting up scrna-longformer working environment..."

# Check if we're in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "scrna_fixed" ]]; then
    echo "❌ Not in scrna_fixed environment (currently in: ${CONDA_DEFAULT_ENV:-none})"
    echo "🔧 Activating scrna_fixed environment..."
    
    # Check if environment exists
    if conda env list | grep -q "scrna_fixed"; then
        echo "✅ Environment exists, activating..."
        source activate scrna_fixed || conda activate scrna_fixed
    else
        echo "❌ Environment 'scrna_fixed' not found!"
        echo "📦 Creating environment..."
        
        conda create -n scrna_fixed python=3.10 -y
        conda activate scrna_fixed
        
        echo "📦 Installing packages..."
        if command -v mamba &> /dev/null; then
            mamba install -c conda-forge scanpy anndata umap-learn python-igraph leidenalg libomp numpy pandas scikit-learn matplotlib pyyaml -y
        else
            conda install -c conda-forge scanpy anndata umap-learn python-igraph leidenalg libomp numpy pandas scikit-learn matplotlib pyyaml -y
        fi
        
        conda install -c pytorch -c conda-forge pytorch torchvision torchaudio -y
        
        echo "✅ Environment created and packages installed!"
    fi
else
    echo "✅ Already in scrna_fixed environment"
fi

# Verify critical packages
echo "🔍 Verifying environment..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import scanpy as sc
    print(f'✅ scanpy: {sc.__version__}')
except ImportError as e:
    print(f'❌ scanpy: {e}')
    exit(1)

try:
    import leidenalg
    print('✅ leidenalg: OK')
except ImportError as e:
    print(f'❌ leidenalg: {e}')
    exit(1)

try:
    import torch
    print(f'✅ torch: {torch.__version__}')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'✅ device: {device}')
except ImportError as e:
    print(f'❌ torch: {e}')
    exit(1)

print('🎉 All packages verified!')
"

echo "✅ Environment setup complete!"
echo ""
echo "📋 Quick commands:"
echo "  Test leiden clustering: PYTHONPATH=./src python -c \"import scanpy as sc; print('leiden test:', hasattr(sc.tl, 'leiden'))\""
echo "  Generate data: PYTHONPATH=./src python scripts/prepare_pbmc3k.py --out data/test.npz"
echo "  Run CV: PYTHONPATH=./src python scripts/run_cv.py --folds 2 --config configs/real_leiden_top500.yaml"
