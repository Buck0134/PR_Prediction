#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "myenv" ]; then
    # The virtual environment does not exist, so create it using python3
    echo "Creating a new virtual environment named myenv..."
    python3 -m venv myenv
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source myenv/bin/activate

# Install required Python packages from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "Environment setup is complete."

echo "You can start the virtual env with the following command"
echo "source myenv/bin/activate"