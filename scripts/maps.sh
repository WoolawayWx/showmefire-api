#!/bin/bash

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate the virtual environment
source ./venv/bin/activate

# Run all Python scripts in the maps folder
for script in ./maps/*.py; do
    echo "Running $script..."
    python "$script"
done

echo "Uploading to CDN..."
python scripts/upload_cdn.py

echo "Generating RSS feed..."
python -m services.rss --add-summary

# Deactivate the virtual environment
deactivate