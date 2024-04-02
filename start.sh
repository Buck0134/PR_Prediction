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

# Define the project's data directory relative to the script location
DATA_DIR="./data"
CSV_FILE="$DATA_DIR/new_pullreq.csv"

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "$CSV_FILE does not exist. Downloading..."
    # Create the data directory if it doesn't exist
    mkdir -p "$DATA_DIR"
    # Use gdown or an equivalent method to download the file into the project's data directory
    # Note: The warning about --id being deprecated suggests you can simply use the file ID without the --id flag
    gdown "14iSHuZqjzwdzyldJZeSkvTs9T1ToXzDi" -O "$CSV_FILE"
fi

echo "Your Data is Now Available"

CLEAN_CSV_FILE="$DATA_DIR/processedData.csv"
if [ ! -f "$CLEAN_CSV_FILE" ]; then
    echo "$CLEAN_CSV_FILE does not exist. Processing..."
    mkdir -p "$DATA_DIR"
    cd preprocess
    python3 process.py
    cd ..
fi

echo "\033[32mCleaned Data is now available at $CLEAN_CSV_FILE\033[0m"

echo "\033[32mYou can start the virtual env with the following command\033[0m"
echo "\033[32msource myenv/bin/activate\033[0m"