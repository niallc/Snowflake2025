#!/bin/bash
# Setup script for Snowflake2025 development environment
# Run this script to activate the virtual environment and ensure editable install

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

# Ensure editable install is present (recommended)
echo "🔗 Ensuring editable install (pip install -e .) ..."
python -c "import hex_ai" >/dev/null 2>&1 || pip install -e .

# Validate import
echo "✅ Validating import..."
python -c "import hex_ai; print('hex_ai import OK')"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "You can now run commands like:"
echo "  python scripts/process_all_trmph.py"
echo "  pytest tests/"
echo "  python scripts/hyperparam_sweep.py"
echo ""
echo "To use this environment in a new shell, run:"
echo "  source hex_ai_env/bin/activate"