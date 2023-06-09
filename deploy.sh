#!/bin/bash

# Set environment name
ENV_NAME="cricket"

# Create new conda environment
conda create --name $ENV_NAME python=3.9 -y

# Activate conda environment
source activate $ENV_NAME

# Install packages
pip install -r requirements.txt

echo "All packages installed in $ENV_NAME successfully!"

chmod +x test.sh

echo "You can test the model by running: bash test.sh"