#!/bin/bash

# Change to the api directory (parent of scripts/)
cd "$(dirname "$0")/.."

# Activate the virtual environment
source venv/bin/activate

# Run the end-of-day data archiving script
python scripts/endOfDay.py

# Run the end-of-day validation report script
python forecast/endOfDayReport.py
