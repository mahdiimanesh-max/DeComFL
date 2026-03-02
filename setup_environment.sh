#!/bin/bash
# DeComFL Environment Setup Script
# This script helps set up the DeComFL environment

set -e

echo "=== DeComFL Environment Setup ==="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ 'uv' is not installed or not in PATH"
    echo ""
    echo "Please install uv first using one of these methods:"
    echo ""
    echo "Option 1 (Recommended - macOS/Linux):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Option 2 (macOS with Homebrew):"
    echo "  brew install uv"
    echo ""
    echo "Option 3 (via pip):"
    echo "  pip install uv"
    echo ""
    echo "After installing, add uv to your PATH:"
    echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\"  # for install.sh method"
    echo ""
    exit 1
fi

echo "✅ 'uv' is installed: $(which uv)"
echo "   Version: $(uv --version)"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

echo "📦 Setting up virtual environment and installing dependencies..."
echo ""

# Install with dev dependencies
echo "Running: uv sync --extra dev"
uv sync --extra dev

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To use the environment, run commands with 'uv run':"
echo "  uv run python decomfl_main.py --help"
echo ""
echo "Or activate the virtual environment:"
echo "  source .venv/bin/activate  # Linux/macOS"
echo "  .venv\\Scripts\\activate     # Windows"
echo ""
