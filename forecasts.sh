#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Run all Python scripts in the maps folder
python forecast/forecastedfiredanger.py


# Deactivate the virtual environment
deactivate