#!/bin/bash
# 🏢 NICEGOLD Enterprise ProjectP - Complete Installation Script
# This script provides complete installation for all dependencies
# Cross-platform compatible (Linux, macOS, Windows with WSL)

echo "🏢 NICEGOLD Enterprise ProjectP - Complete Installation"
echo "======================================================"
echo "🚀 Starting complete dependency installation..."
echo ""

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8

# Update pip first
echo "📦 Updating pip..."
python -m pip install --upgrade pip

# Install rich for better UI
echo "🎨 Installing rich for better terminal UI..."
pip install rich

# Run installation menu for interactive setup
echo "🎯 Starting interactive installation menu..."
python installation_menu.py

# Additional critical packages installation
echo ""
echo "🔧 Installing additional critical packages..."
pip install --upgrade QuantLib python-dotenv pycryptodome
pip install "dask[complete]" "ray[default]" ipython google-cloud-storage azure-storage-blob
pip install ta  # Alternative to TA-Lib

# Verify installation
echo ""
echo "🔍 Verifying critical packages..."
python -c "
import sys
packages = ['QuantLib', 'dotenv', 'Crypto', 'dask', 'ray', 'IPython', 'google.cloud.storage', 'azure.storage.blob', 'ta']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}: OK')
    except ImportError as e:
        print(f'❌ {pkg}: FAILED')
        failed.append(pkg)
        
if failed:
    print(f'\\n⚠️ {len(failed)} packages failed to install properly')
    sys.exit(1)
else:
    print('\\n🎉 All critical packages installed successfully!')
"

echo ""
echo "✅ Installation script completed!"
echo "📋 Run 'python check_installation.py' to verify installation"
echo "🚀 Then run 'python ProjectP.py' to start the application"
