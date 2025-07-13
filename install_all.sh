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

# Check if user wants interactive or automatic installation
echo ""
echo "📋 Installation Options:"
echo "1. 🚀 Ultimate Auto Installation (Recommended)"
echo "2. 🎯 Interactive Installation Menu"
echo "3. 📦 Basic Requirements Installation"
echo ""

read -p "Select option (1-3, default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo "🚀 Starting Ultimate Auto Installation..."
        python ultimate_auto_installer.py
        ;;
    2)
        echo "🎯 Starting Interactive Installation Menu..."
        python installation_menu.py
        ;;
    3)
        echo "📦 Installing basic requirements..."
        if [ -f "requirements_complete.txt" ]; then
            pip install -r requirements_complete.txt
        elif [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        else
            echo "❌ No requirements file found"
            exit 1
        fi
        ;;
    *)
        echo "❌ Invalid option. Using Ultimate Auto Installation..."
        python ultimate_auto_installer.py
        ;;
esac

# Additional critical packages installation
echo ""
echo "🔧 Installing additional critical packages..."
pip install --upgrade QuantLib python-dotenv pycryptodome
pip install "dask[complete]" "ray[default]" ipython google-cloud-storage azure-storage-blob
pip install ta  # Alternative to TA-Lib

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python check_installation.py

echo ""
echo "✅ Installation script completed!"
echo "📋 Run 'python check_installation.py' to verify installation again"
echo "🚀 Then run 'python ProjectP.py' to start the application"
