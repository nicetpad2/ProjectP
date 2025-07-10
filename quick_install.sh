#!/bin/bash
# NICEGOLD ProjectP - Quick Installation Script
# Installs all required Python libraries for the project

echo "🚀 NICEGOLD ProjectP - Quick Installation"
echo "========================================"
echo "Installing all required Python libraries..."
echo

# Update pip first
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip setuptools wheel

echo
echo "📦 Installing core dependencies from requirements.txt..."
python3 -m pip install -r requirements.txt

echo
echo "📦 Installing additional optional packages..."
python3 -m pip install imbalanced-learn PyWavelets

echo
echo "🧪 Running installation check..."
python3 check_installation.py

echo
echo "✅ Installation complete!"
echo "🚀 You can now run: python3 ProjectP.py"
