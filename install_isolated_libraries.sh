#!/bin/bash
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - ISOLATED INSTALLATION SYSTEM
ระบบติดตั้งไลบรารี่แบบแยกดิสก์ สำหรับป้องกันการขัดแย้งและประหยัดพื้นที่

📋 Features:
- ✅ แยกการติดตั้งออกจากดิสก์หลัก
- ✅ ใช้ Virtual Environment แยกต่างหาก
- ✅ จัดการ dependencies อัตโนมัติ
- ✅ ตรวจสอบ compatibility
- ✅ Production-ready configuration
"""

set -e  # Exit on any error

# 🎯 Configuration
PROJECT_NAME="NICEGOLD_ProjectP"
PROJECT_ROOT="/mnt/data/projects/ProjectP"
INSTALL_TARGET="/home/ACER/.cache/nicegold_env"  # ใช้ home cache (แยกดิสก์)
VENV_NAME="nicegold_enterprise_env"
PYTHON_VERSION="python3"

# 🎨 Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 📋 Functions
print_header() {
    echo -e "${PURPLE}================================================================================================${NC}"
    echo -e "${CYAN}🏢 NICEGOLD ENTERPRISE PROJECTP - ISOLATED LIBRARY INSTALLATION${NC}"
    echo -e "${PURPLE}================================================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    print_step "Checking Prerequisites..."
    
    # Check Python
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    python_ver=$($PYTHON_VERSION --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
    print_success "Python $python_ver found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 not found. Please install pip"
        exit 1
    fi
    print_success "pip3 found"
    
    # Check disk space
    available_space=$(df /tmp | tail -1 | awk '{print $4}')
    required_space=2097152  # 2GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space in /tmp. Available: ${available_space}KB, Required: ${required_space}KB"
        print_warning "Continuing anyway, but consider freeing up space..."
    else
        print_success "Sufficient disk space available: ${available_space}KB"
    fi
}

create_isolated_environment() {
    print_step "Creating Isolated Virtual Environment..."
    
    # Remove existing environment if present
    if [ -d "$INSTALL_TARGET/$VENV_NAME" ]; then
        print_warning "Removing existing environment..."
        rm -rf "$INSTALL_TARGET/$VENV_NAME"
    fi
    
    # Create target directory
    mkdir -p "$INSTALL_TARGET"
    
    # Create virtual environment in separate disk
    $PYTHON_VERSION -m venv "$INSTALL_TARGET/$VENV_NAME"
    print_success "Virtual environment created at: $INSTALL_TARGET/$VENV_NAME"
    
    # Activate environment
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_success "pip upgraded to latest version"
}

install_core_dependencies() {
    print_step "Installing Core Dependencies..."
    
    # Ensure we're in the virtual environment
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    # Install core packages with specific versions (NumPy 1.x for SHAP compatibility)
    print_step "Installing NumPy 1.26.4 (SHAP compatible)..."
    pip install numpy==1.26.4
    
    print_step "Installing core data science libraries..."
    pip install pandas==2.2.3
    pip install scikit-learn==1.5.2
    pip install scipy==1.13.1
    pip install joblib==1.4.2
    
    print_success "Core dependencies installed successfully"
}

install_ml_dependencies() {
    print_step "Installing Machine Learning Dependencies..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    # TensorFlow (CPU only)
    print_step "Installing TensorFlow 2.17.0 (CPU-only)..."
    pip install tensorflow==2.17.0
    
    # PyTorch (CPU only)
    print_step "Installing PyTorch 2.4.1 (CPU-only)..."
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    
    # Reinforcement Learning
    print_step "Installing Reinforcement Learning libraries..."
    pip install stable-baselines3==2.3.0
    pip install gymnasium==0.29.1
    
    print_success "Machine Learning dependencies installed successfully"
}

install_advanced_features() {
    print_step "Installing Advanced Feature Selection & Optimization..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    # SHAP (requires NumPy 1.x)
    print_step "Installing SHAP 0.45.0..."
    pip install shap==0.45.0
    
    # Optuna
    print_step "Installing Optuna 3.5.0..."
    pip install optuna==3.5.0
    
    print_success "Advanced features installed successfully"
}

install_data_processing() {
    print_step "Installing Data Processing Libraries..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    pip install PyYAML==6.0.2
    pip install PyWavelets==1.8.0
    pip install imbalanced-learn==0.13.0
    pip install ta==0.11.0
    pip install opencv-python-headless==4.11.0.0
    pip install Pillow==11.2.1
    
    print_success "Data processing libraries installed successfully"
}

install_visualization() {
    print_step "Installing Visualization Libraries..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    pip install matplotlib>=3.5.0
    pip install seaborn>=0.11.0
    pip install plotly>=5.0.0
    
    print_success "Visualization libraries installed successfully"
}

install_development_tools() {
    print_step "Installing Development Tools..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    pip install pytest>=6.0.0
    pip install black>=22.0.0
    pip install flake8>=4.0.0
    
    print_success "Development tools installed successfully"
}

create_activation_script() {
    print_step "Creating Activation Script..."
    
    # Create activation script in project root
    cat > "$PROJECT_ROOT/activate_nicegold_env.sh" << EOF
#!/bin/bash
# 🏢 NICEGOLD Enterprise ProjectP - Environment Activation Script
# Activates the isolated Python environment for NICEGOLD ProjectP

VENV_PATH="$INSTALL_TARGET/$VENV_NAME"

if [ ! -d "\$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: \$VENV_PATH"
    echo "📋 Please run the installation script first: ./install_isolated_libraries.sh"
    exit 1
fi

echo "🚀 Activating NICEGOLD Enterprise Environment..."
echo "📍 Environment Path: \$VENV_PATH"

# Activate the environment
source "\$VENV_PATH/bin/activate"

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
EOF

    chmod +x "$PROJECT_ROOT/activate_nicegold_env.sh"
    print_success "Activation script created: $PROJECT_ROOT/activate_nicegold_env.sh"
}

verify_installation() {
    print_step "Verifying Installation..."
    
    source "$INSTALL_TARGET/$VENV_NAME/bin/activate"
    
    # Test critical imports
    python3 -c "
import sys
print('🐍 Python version:', sys.version)
print('📍 Environment path:', sys.prefix)

# Test core libraries
try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy import failed:', e)

try:
    import pandas as pd
    print('✅ Pandas:', pd.__version__)
except ImportError as e:
    print('❌ Pandas import failed:', e)

try:
    import sklearn
    print('✅ Scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('❌ Scikit-learn import failed:', e)

try:
    import tensorflow as tf
    print('✅ TensorFlow:', tf.__version__)
except ImportError as e:
    print('❌ TensorFlow import failed:', e)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except ImportError as e:
    print('❌ PyTorch import failed:', e)

try:
    import shap
    print('✅ SHAP:', shap.__version__)
except ImportError as e:
    print('❌ SHAP import failed:', e)

try:
    import optuna
    print('✅ Optuna:', optuna.__version__)
except ImportError as e:
    print('❌ Optuna import failed:', e)

print('\\n🎉 Verification complete!')
"
    
    print_success "Installation verification completed!"
}

print_final_instructions() {
    echo ""
    echo -e "${PURPLE}================================================================================================${NC}"
    echo -e "${GREEN}🎉 NICEGOLD ENTERPRISE PROJECTP - INSTALLATION COMPLETE!${NC}"
    echo -e "${PURPLE}================================================================================================${NC}"
    echo ""
    echo -e "${CYAN}📋 How to use:${NC}"
    echo -e "${YELLOW}1. Activate environment:${NC}"
    echo -e "   ${BLUE}source activate_nicegold_env.sh${NC}"
    echo ""
    echo -e "${YELLOW}2. Run ProjectP:${NC}"
    echo -e "   ${BLUE}python ProjectP.py${NC}"
    echo ""
    echo -e "${YELLOW}3. Deactivate when done:${NC}"
    echo -e "   ${BLUE}deactivate${NC}"
    echo ""
    echo -e "${CYAN}📍 Environment Location:${NC}"
    echo -e "   ${BLUE}$INSTALL_TARGET/$VENV_NAME${NC}"
    echo ""
    echo -e "${CYAN}🔧 Activation Script:${NC}"
    echo -e "   ${BLUE}$PROJECT_ROOT/activate_nicegold_env.sh${NC}"
    echo ""
    echo -e "${GREEN}🚀 Ready for production trading!${NC}"
    echo ""
}

# 🚀 Main Execution
main() {
    print_header
    
    cd "$PROJECT_ROOT"
    
    check_prerequisites
    create_isolated_environment
    install_core_dependencies
    install_ml_dependencies
    install_advanced_features
    install_data_processing
    install_visualization
    install_development_tools
    create_activation_script
    verify_installation
    print_final_instructions
}

# Execute main function
main "$@"
