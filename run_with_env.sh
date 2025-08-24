#!/bin/bash
# run_with_env.sh - Run commands with the correct environment activated

# Source the setup script to ensure correct environment
source ./setup_environment.sh

# Run the provided command
echo "🏃 Running: $@"
echo ""
"$@"
