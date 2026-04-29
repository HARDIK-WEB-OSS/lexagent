#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         LEXAGENT — Local Contract Intelligence           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✓ Python $PYTHON_VERSION"
else
    echo "✗ Python 3.11+ required"
    exit 1
fi

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
    if curl -s http://localhost:11434/api/tags | grep -q "mistral"; then
        echo "✓ mistral:7b available"
    else
        echo "⚠  mistral:7b not found. Pulling now..."
        ollama pull mistral:7b
    fi
else
    echo "⚠  Ollama not running. Summaries will use template fallback."
fi

if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR available"
else
    echo "⚠  Tesseract not installed. Scanned PDFs will not work."
    echo "   Install: sudo apt-get install tesseract-ocr"
fi

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at .venv/"
fi

source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

mkdir -p uploads models

echo ""
echo "Starting LexAgent server..."
echo "→ Open: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop."
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
