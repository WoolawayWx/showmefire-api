#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Run all Python scripts in the maps folder
for script in ./maps/*.py; do
    echo "Running $script..."
    python "$script"
done

# Deactivate the virtual environment
deactivate