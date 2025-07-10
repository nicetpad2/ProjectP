#!/bin/bash
# 🏢 NICEGOLD Enterprise ProjectP - Environment Activation Script
# Activates the isolated Python environment for NICEGOLD ProjectP

VENV_PATH="/home/ACER/.cache/nicegold_env/nicegold_enterprise_env"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "📋 Please run the installation script first: ./install_isolated_libraries.sh"
    exit 1
fi

echo "🚀 Activating NICEGOLD Enterprise Environment..."
echo "📍 Environment Path: $VENV_PATH"

# Activate the environment
source "$VENV_PATH/bin/activate"

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES='-1'
export TF_CPP_MIN_LOG_LEVEL='3'
export TF_ENABLE_ONEDNN_OPTS='0'
export PYTHONIOENCODING='utf-8'
export TF_XLA_FLAGS='--tf_xla_enable_xla_devices=false'
export XLA_FLAGS='--xla_gpu_cuda_data_dir=""'

echo "✅ Environment activated successfully!"
echo "🎯 You can now run: python ProjectP.py"
echo ""
echo "📋 To deactivate: deactivate"
echo "🔄 To reactivate: source activate_nicegold_env.sh"
