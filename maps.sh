#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Run all Python scripts in the maps folder
for script in ./maps/*.py; do
    echo "Running $script..."
    python "$script"
done

python cdnupload.py

python -c "from rss_feed import generate_rss_feed; open('feed.xml', 'w').write(generate_rss_feed())"


# Deactivate the virtual environment
deactivate