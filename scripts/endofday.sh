#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the end-of-day data archiving script
python scripts/endOfDay.py

# Run the end-of-day validation report script
python forecast/endOfDayReport.py
