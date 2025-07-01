#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - ENVIRONMENT MANAGER
ระบบจัดการ Environment แบบ Advanced พร้อม Health Check

📋 Features:
- ✅ ตรวจสอบสถานะ Environment
- ✅ วิเคราะห์ Dependencies  
- ✅ ซ่อมแซม Environment อัตโนมัติ
- ✅ สร้างรายงานสถานะ
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class NicegoldEnvironmentManager:
    """ระบบจัดการ Environment สำหรับ NICEGOLD ProjectP"""
    
    def __init__(self):
        self.project_root = Path("/mnt/data/projects/ProjectP")
        self.env_root = Path("/home/ACER/.cache/nicegold_env")
        self.venv_name = "nicegold_enterprise_env"
        self.venv_path = self.env_root / self.venv_name
        self.activation_script = self.project_root / "activate_nicegold_env.sh"
        
        # Required packages with versions
        self.required_packages = {
            'numpy': '1.26.4',
            'pandas': '2.2.3', 
            'scikit-learn': '1.5.2',
            'tensorflow': '2.17.0',
            'torch': '2.4.1',
            'shap': '0.45.0',
            'optuna': '3.5.0',
            'PyYAML': '6.0.2',
            'matplotlib': '3.5.0',
            'plotly': '5.0.0'
        }
    
    def print_header(self):
        """แสดงหัวข้อ"""
        print("=" * 80)
        print("🏢 NICEGOLD ENTERPRISE PROJECTP - ENVIRONMENT MANAGER")
        print("=" * 80)
        print()
    
    def check_environment_exists(self) -> bool:
        """ตรวจสอบว่า Environment มีอยู่หรือไม่"""
        return self.venv_path.exists() and (self.venv_path / "bin" / "python").exists()
    
    def check_package_installed(self, package_name: str, required_version: Optional[str] = None) -> Tuple[bool, str]:
        """ตรวจสอบว่า package ติดตั้งแล้วหรือไม่"""
        try:
            result = subprocess.run([
                str(self.venv_path / "bin" / "python"), "-c",
                f"import {package_name.replace('-', '_')}; print({package_name.replace('-', '_')}.__version__)"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                installed_version = result.stdout.strip()
                if required_version:
                    # Simple version comparison
                    if installed_version.startswith(required_version.split('.')[0]):
                        return True, installed_version
                    else:
                        return False, f"Version mismatch: {installed_version} (required: {required_version})"
                return True, installed_version
            else:
                return False, "Not installed"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_environment_status(self) -> Dict:
        """ได้รับสถานะของ Environment"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'environment_exists': self.check_environment_exists(),
            'activation_script_exists': self.activation_script.exists(),
            'packages': {},
            'health_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        if not status['environment_exists']:
            status['issues'].append("Virtual environment not found")
            status['recommendations'].append("Run install_isolated_libraries.sh to create environment")
            return status
        
        # Check packages
        total_packages = len(self.required_packages)
        working_packages = 0
        
        for package, version in self.required_packages.items():
            is_installed, details = self.check_package_installed(package, version)
            status['packages'][package] = {
                'installed': is_installed,
                'details': details,
                'required_version': version
            }
            
            if is_installed:
                working_packages += 1
            else:
                status['issues'].append(f"{package}: {details}")
        
        # Calculate health score
        status['health_score'] = int((working_packages / total_packages) * 100)
        
        # Recommendations
        if status['health_score'] < 80:
            status['recommendations'].append("Reinstall environment for better compatibility")
        elif status['health_score'] < 100:
            status['recommendations'].append("Update missing packages")
        
        if not status['activation_script_exists']:
            status['issues'].append("Activation script missing")
            status['recommendations'].append("Recreate activation script")
        
        return status
    
    def create_status_report(self) -> str:
        """สร้างรายงานสถานะ"""
        status = self.get_environment_status()
        
        report = f"""
🏢 NICEGOLD ENTERPRISE PROJECTP - ENVIRONMENT STATUS REPORT
Generated: {status['timestamp']}

📊 OVERALL HEALTH SCORE: {status['health_score']}%

🔍 ENVIRONMENT CHECK:
✅ Environment Exists: {'Yes' if status['environment_exists'] else 'No'}
✅ Activation Script: {'Yes' if status['activation_script_exists'] else 'No'}

📦 PACKAGE STATUS:
"""
        
        for package, info in status['packages'].items():
            status_icon = "✅" if info['installed'] else "❌"
            report += f"{status_icon} {package}: {info['details']}\n"
        
        if status['issues']:
            report += f"\n⚠️ ISSUES FOUND ({len(status['issues'])}):\n"
            for issue in status['issues']:
                report += f"   - {issue}\n"
        
        if status['recommendations']:
            report += f"\n💡 RECOMMENDATIONS:\n"
            for rec in status['recommendations']:
                report += f"   - {rec}\n"
        
        # Health assessment
        if status['health_score'] >= 90:
            report += "\n🎉 EXCELLENT: Environment is production-ready!"
        elif status['health_score'] >= 70:
            report += "\n👍 GOOD: Environment is mostly functional"
        elif status['health_score'] >= 50:
            report += "\n⚠️ FAIR: Environment needs attention"
        else:
            report += "\n❌ POOR: Environment requires reinstallation"
        
        return report
    
    def save_status_report(self, filename: str = None) -> str:
        """บันทึกรายงานสถานะ"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"environment_status_{timestamp}.txt"
        
        report_path = self.project_root / filename
        report = self.create_status_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(report_path)
    
    def fix_activation_script(self) -> bool:
        """ซ่อมแซมไฟล์ activation script"""
        try:
            script_content = f"""#!/bin/bash
# 🏢 NICEGOLD Enterprise ProjectP - Environment Activation Script
# Activates the isolated Python environment for NICEGOLD ProjectP

VENV_PATH="{self.venv_path}"

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
"""
            
            with open(self.activation_script, 'w') as f:
                f.write(script_content)
            
            os.chmod(self.activation_script, 0o755)
            return True
        except Exception as e:
            print(f"❌ Failed to fix activation script: {e}")
            return False
    
    def quick_fix(self) -> bool:
        """ซ่อมแซมปัญหาเบื้องต้น"""
        print("🔧 Running Quick Fix...")
        
        fixed = False
        
        # Fix activation script if missing
        if not self.activation_script.exists():
            print("🔧 Fixing activation script...")
            if self.fix_activation_script():
                print("✅ Activation script fixed")
                fixed = True
            else:
                print("❌ Failed to fix activation script")
        
        return fixed

def main():
    """ฟังก์ชันหลัก"""
    manager = NicegoldEnvironmentManager()
    manager.print_header()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            print(manager.create_status_report())
            
        elif command == "report":
            report_path = manager.save_status_report()
            print(f"📋 Status report saved to: {report_path}")
            
        elif command == "fix":
            if manager.quick_fix():
                print("✅ Quick fix completed")
            else:
                print("❌ Quick fix failed")
                
        elif command == "health":
            status = manager.get_environment_status()
            print(f"🏥 Environment Health Score: {status['health_score']}%")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: status, report, fix, health")
    else:
        # Default: show status
        print(manager.create_status_report())

if __name__ == "__main__":
    main()
