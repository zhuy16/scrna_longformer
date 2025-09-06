#!/bin/bash
# verify_environment.sh - Quick check that we're in the correct environment

echo "🔍 Environment Verification"
echo "=========================="

# Check current environment
if [[ "$CONDA_DEFAULT_ENV" == "scrna_fixed" ]]; then
    echo "✅ Environment: $CONDA_DEFAULT_ENV (CORRECT)"
else
    echo "❌ Environment: ${CONDA_DEFAULT_ENV:-none} (SHOULD BE: scrna_fixed)"
    echo "   Run: conda activate scrna_fixed"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "✅ Python: $PYTHON_VERSION"

# Check critical packages
echo "🔍 Package verification:"

python -c "
import sys
errors = 0

try:
    import scanpy as sc
    print('  ✅ scanpy:', sc.__version__)
except ImportError as e:
    print('  ❌ scanpy:', e)
    errors += 1

try:
    import leidenalg
    print('  ✅ leidenalg: OK')
except ImportError as e:
    print('  ❌ leidenalg:', e)
    errors += 1

try:
    import torch
    print('  ✅ torch:', torch.__version__)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('  ✅ device:', device)
except ImportError as e:
    print('  ❌ torch:', e)
    errors += 1

if errors == 0:
    print('\\n🎉 All packages verified!')
    sys.exit(0)
else:
    print(f'\\n❌ {errors} package errors found')
    print('   Run: ./setup_environment.sh')
    sys.exit(1)
"
