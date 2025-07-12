
#!/bin/bash
# This script sets up the environment for the local voice processing project.

# Create a conda environment
conda create -n local_voice_processing python=3.11 -y

# Activate the environment
conda activate local_voice_processing

# Install dependencies
pip install -r requirements.txt

echo "Environment 'local_voice_processing' created and activated."
