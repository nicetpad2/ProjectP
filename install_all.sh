#!/bin/bash
# 🚀 NICEGOLD ProjectP-1 - Complete Installation Script
# Enterprise-Grade Installation for Linux/macOS
# Date: July 9, 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}${1}${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..80})${NC}"
}

print_step() {
    echo -e "\n${BLUE}📋 Step ${1}: ${2}${NC}"
}

print_success() {
    echo -e "${GREEN}✅ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ ${1}${NC}"
}

print_error() {
    echo -e "${RED}❌ ${1}${NC}"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_success "Python version: $PYTHON_VERSION"
}

# Check if pip is available
check_pip() {
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip is not installed. Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm get-pip.py
    fi
    print_success "pip is available"
}

# Install system dependencies (if needed)
install_system_deps() {
    print_step 1 "Installing system dependencies"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            print_warning "Detected Ubuntu/Debian - Installing system packages..."
            sudo apt-get update
            sudo apt-get install -y python3-dev python3-pip build-essential
        elif command -v yum &> /dev/null; then
            print_warning "Detected CentOS/RHEL - Installing system packages..."
            sudo yum install -y python3-devel python3-pip gcc gcc-c++
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            print_warning "Detected macOS - Installing system packages..."
            brew install python3
        fi
    fi
    
    print_success "System dependencies installed"
}

# Main installation function
main() {
    print_header "🚀 NICEGOLD ProjectP-1 - Complete Installation"
    echo "📅 Installation Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "🖥️ Operating System: $OSTYPE"
    echo "📁 Working Directory: $(pwd)"
    
    # Check prerequisites
    check_python
    check_pip
    
    # Install system dependencies
    install_system_deps
    
    # Run Python installation scripts
    print_step 2 "Running Python dependency installation"
    
    # Upgrade pip first
    print_warning "Upgrading pip to latest version..."
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Install from requirements.txt if exists
    if [ -f "requirements.txt" ]; then
        print_warning "Installing from requirements.txt..."
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
    
    # Run comprehensive installation script
    if [ -f "install_complete_dependencies.py" ]; then
        print_warning "Running comprehensive dependency installation..."
        $PYTHON_CMD install_complete_dependencies.py
    fi
    
    # Run library installation script
    if [ -f "install_all_libraries.py" ]; then
        print_warning "Running library installation..."
        $PYTHON_CMD install_all_libraries.py
    fi
    
    print_step 3 "Installation Verification"
    
    # Test basic imports
    print_warning "Testing critical imports..."
    $PYTHON_CMD -c "import numpy, pandas, sklearn, tensorflow, torch; print('✅ Core packages imported successfully')" || print_error "Some core packages failed to import"
    
    print_step 4 "Installation Complete"
    print_success "🎉 NICEGOLD ProjectP-1 installation completed!"
    print_success "🚀 You can now run: python ProjectP.py"
    print_success "🌊 Select Menu 1 for Elliott Wave Analysis"
    
    # Create installation log
    echo "Installation completed at $(date)" > installation.log
    echo "Python version: $PYTHON_VERSION" >> installation.log
    echo "Installation directory: $(pwd)" >> installation.log
    
    print_success "📝 Installation log saved to installation.log"
}

# Run main function
main "$@"
