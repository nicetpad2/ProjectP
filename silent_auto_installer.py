#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD ENTERPRISE PROJECTP - SILENT AUTO INSTALLER
🏢 Complete Silent Installation System for CI/CD and Automated Setup

This script provides silent installation for the entire NICEGOLD ProjectP system
without any user interaction, perfect for automated deployment.

Author: NICEGOLD Enterprise ProjectP Team
Version: v3.0 Silent Edition
Date: July 12, 2025
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

class SilentInstaller:
    """
    🚀 NICEGOLD Silent Auto Installer
    Complete automated installation without user interaction
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.install_log = []
        
    def log_message(self, message: str, level: str = "INFO"):
        """Log installation message"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.install_log.append(log_entry)
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: list, description: str = "") -> bool:
        """Run command and log results"""
        try:
            self.log_message(f"Executing: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.log_message(f"✅ {description} completed successfully", "SUCCESS")
                return True
            else:
                self.log_message(f"❌ {description} failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_message(f"⏰ {description} timed out", "ERROR")
            return False
        except Exception as e:
            self.log_message(f"❌ {description} error: {e}", "ERROR")
            return False
    
    def install_packages(self, packages: list, description: str = "packages") -> bool:
        """Install multiple packages"""
        self.log_message(f"📦 Installing {description}...")
        
        success_count = 0
        total_count = len(packages)
        
        for package in packages:
            success = self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            )
            if success:
                success_count += 1
        
        self.log_message(f"📊 {description}: {success_count}/{total_count} packages installed successfully")
        return success_count >= (total_count * 0.8)  # 80% success rate required
    
    def install_all_dependencies(self) -> bool:
        """Install all dependencies silently"""
        self.log_message("🚀 Starting silent installation of all dependencies")
        
        # Step 1: Update pip
        self.log_message("📦 Step 1: Updating pip...")
        self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "pip update")
        
        # Step 2: Essential packages
        essential_packages = [
            "wheel>=0.37.0",
            "setuptools>=60.0.0",
            "rich>=12.0.0",
            "colorama>=0.4.6",
            "psutil>=5.8.0",
            "PyYAML>=6.0.0",
            "joblib>=1.4.2"
        ]
        self.install_packages(essential_packages, "essential packages")
        
        # Step 3: Fix NumPy/SHAP compatibility
        self.log_message("🔧 Step 3: Fixing NumPy/SHAP compatibility...")
        self.run_command([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], "NumPy 1.26.4")
        self.run_command([sys.executable, "-m", "pip", "install", "shap>=0.48.0"], "SHAP")
        
        # Step 4: Core data science packages
        core_packages = [
            "pandas>=2.2.3",
            "scipy>=1.14.1",
            "scikit-learn>=1.5.2",
            "optuna>=4.4.0"
        ]
        self.install_packages(core_packages, "core data science packages")
        
        # Step 5: Machine learning packages
        ml_packages = [
            "tensorflow>=2.18.0",
            "torch>=2.6.0", 
            "torchvision>=0.21.0",
            "xgboost>=2.1.4",
            "lightgbm>=4.5.0"
        ]
        self.install_packages(ml_packages, "machine learning packages")
        
        # Step 6: Reinforcement learning
        rl_packages = [
            "gymnasium>=1.1.1",
            "stable-baselines3>=2.6.0"
        ]
        self.install_packages(rl_packages, "reinforcement learning packages")
        
        # Step 7: Financial packages
        financial_packages = [
            "yfinance>=0.2.38",
            "ta>=0.11.0",
            "quantlib>=1.36",
            "statsmodels>=0.14.4"
        ]
        self.install_packages(financial_packages, "financial analysis packages")
        
        # Step 8: Visualization packages
        viz_packages = [
            "matplotlib>=3.9.2",
            "seaborn>=0.13.2",
            "plotly>=5.24.1"
        ]
        self.install_packages(viz_packages, "visualization packages")
        
        # Step 9: Enterprise packages
        enterprise_packages = [
            "alembic>=1.16.2",
            "cryptography>=44.0.0",
            "sqlalchemy>=2.0.36",
            "h5py>=3.12.1",
            "python-dotenv>=1.0.1"
        ]
        self.install_packages(enterprise_packages, "enterprise packages")
        
        # Step 10: Install from requirements file if available
        self.install_from_requirements()
        
        return True
    
    def install_from_requirements(self) -> bool:
        """Install from requirements file"""
        requirements_files = [
            "requirements_complete.txt",
            "requirements.txt"
        ]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.log_message(f"📋 Installing from {req_file}...")
                success = self.run_command(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                    f"requirements from {req_file}"
                )
                if success:
                    return True
        
        self.log_message("⚠️ No requirements file found or installation failed", "WARNING")
        return False
    
    def verify_critical_packages(self) -> bool:
        """Verify that critical packages are installed"""
        critical_packages = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
            'shap', 'optuna', 'rich', 'colorama', 'psutil', 'joblib'
        ]
        
        self.log_message("🔍 Verifying critical packages...")
        
        failed_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
                self.log_message(f"✅ {package} verified", "SUCCESS")
            except ImportError:
                failed_packages.append(package)
                self.log_message(f"❌ {package} verification failed", "ERROR")
        
        if failed_packages:
            self.log_message(f"❌ {len(failed_packages)} critical packages failed: {failed_packages}", "ERROR")
            return False
        else:
            self.log_message("🎉 All critical packages verified successfully!", "SUCCESS")
            return True
    
    def create_installation_report(self) -> str:
        """Create installation report"""
        report = {
            'installation_type': 'silent_auto_installer',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': os.name,
            'install_log': self.install_log,
            'total_steps': len([log for log in self.install_log if log['level'] == 'SUCCESS']),
            'total_errors': len([log for log in self.install_log if log['level'] == 'ERROR'])
        }
        
        report_file = self.project_root / f"silent_installation_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log_message(f"📋 Installation report saved: {report_file.name}", "SUCCESS")
            return str(report_file)
        except Exception as e:
            self.log_message(f"❌ Could not save installation report: {e}", "ERROR")
            return ""
    
    def run_silent_installation(self) -> bool:
        """Run complete silent installation"""
        start_time = time.time()
        
        print("🚀 NICEGOLD ENTERPRISE PROJECTP - SILENT AUTO INSTALLER")
        print("=" * 60)
        print("🔄 Starting automated installation (no user interaction required)...")
        print("")
        
        try:
            # Install all dependencies
            self.install_all_dependencies()
            
            # Verify installation
            verification_success = self.verify_critical_packages()
            
            # Create report
            report_file = self.create_installation_report()
            
            # Calculate duration
            duration = time.time() - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            print("")
            print("=" * 60)
            if verification_success:
                print("🎉 SILENT INSTALLATION COMPLETED SUCCESSFULLY!")
                print(f"⏰ Installation time: {minutes}m {seconds}s")
                print("✅ All critical packages verified")
                print("")
                print("🚀 Next steps:")
                print("   1. Run: python check_installation.py")
                print("   2. Run: python ProjectP.py")
                print("   3. Enjoy AI-powered trading! 🎯")
            else:
                print("⚠️ SILENT INSTALLATION COMPLETED WITH ISSUES")
                print(f"⏰ Installation time: {minutes}m {seconds}s")
                print("🔧 Some critical packages may need manual installation")
                print("")
                print("🚀 Recommended actions:")
                print("   1. Run: python check_installation.py")
                print("   2. Run: python installation_menu.py")
                print("   3. Check installation report for details")
            
            if report_file:
                print(f"📋 Installation report: {Path(report_file).name}")
            
            return verification_success
            
        except Exception as e:
            self.log_message(f"❌ Critical installation error: {e}", "ERROR")
            print(f"\n❌ Installation failed: {e}")
            return False


def main():
    """Main function"""
    try:
        installer = SilentInstaller()
        success = installer.run_silent_installation()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
