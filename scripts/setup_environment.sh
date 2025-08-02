#!/bin/bash
# Setup script for Snowflake2025 development environment
# Run this script to activate the virtual environment and set PYTHONPATH

set -e  # Exit on any error

echo "🔧 Setting up Snowflake2025 development environment..."

# Check if virtual environment exists
if [ ! -d "hex_ai_env" ]; then
    echo "❌ Virtual environment 'hex_ai_env' not found!"
    echo "Please create it first: python -m venv hex_ai_env"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source hex_ai_env/bin/activate

# Set PYTHONPATH
echo "🔗 Setting PYTHONPATH=."
export PYTHONPATH=.

# Validate environment
echo "✅ Validating environment..."
python scripts/validate_environment.py

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "You can now run commands like:"
echo "  python scripts/process_all_trmph.py"
echo "  python -m pytest tests/"
echo "  python scripts/hyperparam_sweep.py"
echo ""
echo "To use this environment in a new shell, run:"
echo "  source hex_ai_env/bin/activate && export PYTHONPATH=." 