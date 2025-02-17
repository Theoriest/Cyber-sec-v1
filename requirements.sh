#!/bin/bash

# Update package lists
sudo apt-get update

# Install pip
sudo apt-get install -y python3-pip

# Install virtualenv
pip3 install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install necessary build tools
sudo apt-get install -y build-essential

# Install Rust (required for building some Python packages)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install SentencePiece
pip install sentencepiece

# Install necessary Python packages with specific versions for compatibility
pip install pandas scikit-learn torch tensorflow flax transformers

# Verify installation
echo "Installed versions:"
pip show pandas | grep Version
pip show scikit-learn | grep Version
pip show torch | grep Version
pip show tensorflow | grep Version
pip show flax | grep Version
pip show transformers | grep Version

echo "All required modules have been installed."

# Deactivate the virtual environment
deactivate