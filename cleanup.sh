#!/bin/bash
# cleanup.sh — Removes generated/trained artifacts after using the app.
# Keeps: source code, config, venv, README, .gitignore

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Cleaning up..."

# Remove training outputs
if [ -d "output" ]; then
    rm -rf output/
    echo "  Removed output/"
fi

# Remove generated dataset
if [ -f "data/tiny_conversations.json" ]; then
    rm -f data/tiny_conversations.json
    echo "  Removed data/tiny_conversations.json"
fi

# Remove Python cache
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
echo "  Removed __pycache__/"

# Remove .pyc files
find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
echo "  Removed *.pyc"

# Remove OS junk
rm -f .DS_Store 2>/dev/null || true

echo ""
echo "Done. Project is clean."
echo ""
echo "To start fresh:"
echo "  source .venv/bin/activate"
echo "  python make_dataset.py"
echo "  python train.py"
