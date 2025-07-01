#!/bin/bash
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - DISK MANAGEMENT UTILITY
เครื่องมือจัดการดิสก์และพื้นที่สำหรับการติดตั้งแยกดิสก์

📋 Features:
- ✅ ตรวจสอบพื้นที่ดิสก์
- ✅ แนะนำตำแหน่งติดตั้ง
- ✅ จัดการ temporary files
- ✅ สร้าง symbolic links
"""

# 🎨 Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}================================================================================================${NC}"
    echo -e "${CYAN}🏢 NICEGOLD ENTERPRISE PROJECTP - DISK MANAGEMENT UTILITY${NC}"
    echo -e "${PURPLE}================================================================================================${NC}"
    echo ""
}

check_disk_space() {
    echo -e "${BLUE}📊 Checking Disk Space...${NC}"
    echo ""
    
    # Check main project directory
    project_disk=$(df /mnt/data/projects/ProjectP | tail -1)
    echo -e "${CYAN}📍 Project Directory (/mnt/data/projects/ProjectP):${NC}"
    echo "$project_disk" | awk '{print "   💾 Available: " int($4/1024) "MB (" int($4/1024/1024) "GB)"}'
    echo "$project_disk" | awk '{print "   📈 Used: " int($3/1024) "MB (" int($3/1024/1024) "GB)"}'
    echo "$project_disk" | awk '{print "   📊 Total: " int($2/1024) "MB (" int($2/1024/1024) "GB)"}'
    echo ""
    
    # Check /tmp directory 
    tmp_disk=$(df /tmp | tail -1)
    echo -e "${CYAN}📍 Temporary Directory (/tmp):${NC}"
    echo "$tmp_disk" | awk '{print "   💾 Available: " int($4/1024) "MB (" int($4/1024/1024) "GB)"}'
    echo "$tmp_disk" | awk '{print "   📈 Used: " int($3/1024) "MB (" int($3/1024/1024) "GB)"}'
    echo "$tmp_disk" | awk '{print "   📊 Total: " int($2/1024) "MB (" int($2/1024/1024) "GB)"}'
    echo ""
    
    # Check /var directory
    var_disk=$(df /var | tail -1)
    echo -e "${CYAN}📍 Variable Directory (/var):${NC}"
    echo "$var_disk" | awk '{print "   💾 Available: " int($4/1024) "MB (" int($4/1024/1024) "GB)"}'
    echo "$var_disk" | awk '{print "   📈 Used: " int($3/1024) "MB (" int($3/1024/1024) "GB)"}'
    echo "$var_disk" | awk '{print "   📊 Total: " int($2/1024) "MB (" int($2/1024/1024) "GB)"}'
    echo ""
    
    # Check home directory
    home_disk=$(df ~ | tail -1)
    echo -e "${CYAN}📍 Home Directory (~):${NC}"
    echo "$home_disk" | awk '{print "   💾 Available: " int($4/1024) "MB (" int($4/1024/1024) "GB)"}'
    echo "$home_disk" | awk '{print "   📈 Used: " int($3/1024) "MB (" int($3/1024/1024) "GB)"}'
    echo "$home_disk" | awk '{print "   📊 Total: " int($2/1024) "MB (" int($2/1024/1024) "GB)"}'
    echo ""
}

recommend_installation_location() {
    echo -e "${BLUE}🎯 Analyzing Best Installation Location...${NC}"
    echo ""
    
    # Required space for installation (approximately 2GB)
    required_space_kb=2097152  # 2GB in KB
    
    # Check available locations
    declare -A locations
    locations["/tmp"]=$(df /tmp | tail -1 | awk '{print $4}')
    locations["/var/tmp"]=$(df /var/tmp | tail -1 | awk '{print $4}')
    locations["$HOME/.cache"]=$(df ~ | tail -1 | awk '{print $4}')
    locations["/opt"]=$(df /opt 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
    
    echo -e "${CYAN}📋 Available Installation Locations:${NC}"
    
    best_location=""
    best_space=0
    
    for location in "${!locations[@]}"; do
        space=${locations[$location]}
        space_mb=$((space / 1024))
        space_gb=$((space_mb / 1024))
        
        if [ "$space" -gt "$required_space_kb" ]; then
            status="${GREEN}✅ Suitable${NC}"
            if [ "$space" -gt "$best_space" ]; then
                best_location="$location"
                best_space="$space"
            fi
        else
            status="${RED}❌ Insufficient${NC}"
        fi
        
        echo -e "   📍 $location: ${space_mb}MB (${space_gb}GB) - $status"
    done
    
    echo ""
    if [ -n "$best_location" ]; then
        echo -e "${GREEN}🏆 Recommended Location: $best_location${NC}"
        echo -e "   💾 Available Space: $((best_space / 1024))MB ($((best_space / 1024 / 1024))GB)"
        echo -e "   📋 Recommended Path: $best_location/nicegold_env"
    else
        echo -e "${RED}⚠️ No suitable location found with sufficient space!${NC}"
        echo -e "   📋 Required: 2GB minimum"
        echo -e "   💡 Consider cleaning up disk space or using external storage"
    fi
    echo ""
}

clean_temporary_files() {
    echo -e "${BLUE}🧹 Cleaning Temporary Files...${NC}"
    echo ""
    
    # Clean Python cache
    echo -e "${YELLOW}🔧 Cleaning Python cache files...${NC}"
    find /mnt/data/projects/ProjectP -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find /mnt/data/projects/ProjectP -name "*.pyc" -type f -delete 2>/dev/null || true
    find /mnt/data/projects/ProjectP -name "*.pyo" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Python cache cleaned${NC}"
    
    # Clean old log files
    echo -e "${YELLOW}🔧 Cleaning old log files...${NC}"
    find /mnt/data/projects/ProjectP/logs -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Old log files cleaned${NC}"
    
    # Clean temporary outputs
    echo -e "${YELLOW}🔧 Cleaning temporary outputs...${NC}"
    find /mnt/data/projects/ProjectP/temp -type f -mtime +1 -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Temporary outputs cleaned${NC}"
    
    # Clean pip cache if it exists
    if [ -d "$HOME/.cache/pip" ]; then
        echo -e "${YELLOW}🔧 Cleaning pip cache...${NC}"
        rm -rf "$HOME/.cache/pip/*" 2>/dev/null || true
        echo -e "${GREEN}✅ Pip cache cleaned${NC}"
    fi
    
    echo -e "${GREEN}🎉 Cleanup completed!${NC}"
    echo ""
}

create_custom_installation_script() {
    local install_path="$1"
    local script_name="install_custom_location.sh"
    
    echo -e "${BLUE}📝 Creating Custom Installation Script...${NC}"
    
    cat > "$script_name" << EOF
#!/bin/bash
# Custom installation script for NICEGOLD ProjectP
# Installation path: $install_path

INSTALL_TARGET="$install_path"
VENV_NAME="nicegold_enterprise_env"
PROJECT_ROOT="/mnt/data/projects/ProjectP"

echo "🚀 Installing NICEGOLD Environment to: \$INSTALL_TARGET"

# Create directory
mkdir -p "\$INSTALL_TARGET"

# Create virtual environment
python3 -m venv "\$INSTALL_TARGET/\$VENV_NAME"

# Activate and install packages
source "\$INSTALL_TARGET/\$VENV_NAME/bin/activate"
pip install --upgrade pip

# Install from requirements.txt
pip install -r "\$PROJECT_ROOT/requirements.txt"

# Create custom activation script
cat > "\$PROJECT_ROOT/activate_custom_env.sh" << 'SCRIPT_EOF'
#!/bin/bash
echo "🚀 Activating NICEGOLD Custom Environment..."
source "$install_path/\$VENV_NAME/bin/activate"
export CUDA_VISIBLE_DEVICES='-1'
export TF_CPP_MIN_LOG_LEVEL='3'
echo "✅ Environment activated!"
SCRIPT_EOF

chmod +x "\$PROJECT_ROOT/activate_custom_env.sh"

echo "✅ Installation completed!"
echo "📋 To activate: source activate_custom_env.sh"
EOF
    
    chmod +x "$script_name"
    echo -e "${GREEN}✅ Custom installation script created: $script_name${NC}"
    echo ""
}

show_disk_usage() {
    echo -e "${BLUE}📊 Detailed Disk Usage Analysis${NC}"
    echo ""
    
    # Show largest directories in project
    echo -e "${CYAN}📍 Largest directories in ProjectP:${NC}"
    du -sh /mnt/data/projects/ProjectP/* 2>/dev/null | sort -hr | head -10
    echo ""
    
    # Show system disk usage
    echo -e "${CYAN}📍 System disk usage:${NC}"
    df -h | grep -E '^/dev|^tmpfs' | head -10
    echo ""
}

main() {
    print_header
    
    case "${1:-help}" in
        "check")
            check_disk_space
            ;;
        "recommend")
            recommend_installation_location
            ;;
        "clean")
            clean_temporary_files
            ;;
        "usage")
            show_disk_usage
            ;;
        "custom")
            if [ -n "$2" ]; then
                create_custom_installation_script "$2"
            else
                echo -e "${RED}❌ Please specify installation path${NC}"
                echo -e "${BLUE}Usage: $0 custom /path/to/install${NC}"
            fi
            ;;
        "all")
            check_disk_space
            recommend_installation_location
            show_disk_usage
            ;;
        *)
            echo -e "${CYAN}📋 Available Commands:${NC}"
            echo -e "   ${BLUE}check${NC}     - Check disk space for all locations"
            echo -e "   ${BLUE}recommend${NC}  - Recommend best installation location"
            echo -e "   ${BLUE}clean${NC}     - Clean temporary files to free space"
            echo -e "   ${BLUE}usage${NC}     - Show detailed disk usage"
            echo -e "   ${BLUE}custom${NC}    - Create custom installation script"
            echo -e "   ${BLUE}all${NC}       - Run all checks"
            echo ""
            echo -e "${YELLOW}💡 Example:${NC}"
            echo -e "   ${BLUE}$0 check${NC}"
            echo -e "   ${BLUE}$0 recommend${NC}"
            echo -e "   ${BLUE}$0 custom /opt/nicegold${NC}"
            ;;
    esac
}

main "$@"
