#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Updating package lists..."
sudo apt-get update -y

echo "Installing Python3, pip, and venv..."
sudo apt-get install -y python3 python3-pip python3-venv build-essential

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install Rust (if not already installed)
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust is already installed, skipping..."
fi

# Create a requirements.txt file
cat <<EOF > requirements.txt
sentencepiece
pandas
scikit-learn
torch
tensorflow
flax
transformers
EOF

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Verify installations
echo "Installed versions:"
for pkg in sentencepiece pandas scikit-learn torch tensorflow flax transformers; do
    pip show "$pkg" | grep Version || echo "$pkg not found!"
done

echo "All required modules have been installed successfully."

# Deactivate the virtual environment
deactivate
echo "Virtual environment deactivated."
