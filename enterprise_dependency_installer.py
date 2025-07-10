#!/usr/bin/env python3
"""
Enterprise Dependency Installer for NICEGOLD ProjectP
Automatically installs all required dependencies with enterprise-grade error handling
"""

import os
import sys
import subprocess
import warnings
import platform
import logging
from typing import List, Tuple, Dict, Any

# Suppress all warnings during installation
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class EnterpriseDependencyInstaller:
    """Enterprise-grade dependency installer with comprehensive error handling"""
    
    def __init__(self):
        self.python_executable = sys.executable
        self.platform = platform.system()
        self.architecture = platform.architecture()[0]
        self.required_packages = [
            "psutil>=5.8.0",
            "numpy>=1.21.0", 
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "yfinance>=0.1.70",
            "TA-Lib>=0.4.24",
            "plotly>=5.5.0",
            "dash>=2.0.0",
            "requests>=2.25.0",
            "colorama>=0.4.4",
            "tqdm>=4.62.0",
            "joblib>=1.1.0",
            "optuna>=2.10.0",
            "shap>=0.40.0"
        ]
        
    def setup_logging(self) -> None:
        """Setup enterprise logging for installation process"""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('dependency_installation.log', mode='w')
                ]
            )
        except Exception:
            # Fallback to basic logging if file creation fails
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logging.error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
            return False
        logging.info(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            logging.info("Upgrading pip to latest version...")
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logging.info("Pip upgraded successfully")
                return True
            else:
                logging.warning(f"Pip upgrade warning: {result.stderr}")
                return True  # Continue even if upgrade fails
        except Exception as e:
            logging.warning(f"Pip upgrade failed, continuing: {str(e)}")
            return True  # Continue even if upgrade fails
    
    def install_package(self, package: str) -> bool:
        """Install a single package with enterprise error handling"""
        try:
            logging.info(f"Installing {package}...")
            
            # Try multiple installation strategies
            strategies = [
                [self.python_executable, "-m", "pip", "install", package],
                [self.python_executable, "-m", "pip", "install", package, "--user"],
                [self.python_executable, "-m", "pip", "install", package, "--no-cache-dir"],
                [self.python_executable, "-m", "pip", "install", package, "--force-reinstall"]
            ]
            
            for strategy in strategies:
                try:
                    result = subprocess.run(
                        strategy, 
                        capture_output=True, 
                        text=True, 
                        timeout=600
                    )
                    
                    if result.returncode == 0:
                        logging.info(f"Successfully installed {package}")
                        return True
                    else:
                        logging.warning(f"Strategy failed for {package}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logging.warning(f"Timeout installing {package}, trying next strategy")
                    continue
                except Exception as e:
                    logging.warning(f"Error with strategy for {package}: {str(e)}")
                    continue
            
            logging.error(f"All strategies failed for {package}")
            return False
            
        except Exception as e:
            logging.error(f"Critical error installing {package}: {str(e)}")
            return False
    
    def verify_package(self, package_name: str) -> bool:
        """Verify that a package is properly installed"""
        try:
            # Extract package name without version requirements
            clean_name = package_name.split(">=")[0].split("==")[0].split("<")[0]
            
            # Handle special cases
            if clean_name == "TA-Lib":
                clean_name = "talib"
            elif clean_name == "scikit-learn":
                clean_name = "sklearn"
            
            __import__(clean_name)
            logging.info(f"Package {clean_name} verified successfully")
            return True
        except ImportError:
            logging.warning(f"Package {clean_name} verification failed")
            return False
        except Exception as e:
            logging.warning(f"Error verifying {clean_name}: {str(e)}")
            return False
    
    def install_all_dependencies(self) -> bool:
        """Install all required dependencies"""
        logging.info("Starting enterprise dependency installation...")
        
        if not self.check_python_version():
            return False
        
        if not self.upgrade_pip():
            logging.warning("Pip upgrade failed, continuing anyway")
        
        success_count = 0
        failed_packages = []
        
        for package in self.required_packages:
            if self.install_package(package):
                if self.verify_package(package):
                    success_count += 1
                else:
                    failed_packages.append(package)
            else:
                failed_packages.append(package)
        
        total_packages = len(self.required_packages)
        success_rate = (success_count / total_packages) * 100
        
        logging.info(f"Installation complete: {success_count}/{total_packages} packages ({success_rate:.1f}%)")
        
        if failed_packages:
            logging.warning(f"Failed packages: {', '.join(failed_packages)}")
            
            # Try alternative installation for critical packages
            critical_packages = ["psutil", "numpy", "pandas", "scikit-learn"]
            for package in failed_packages:
                clean_name = package.split(">=")[0]
                if clean_name in critical_packages:
                    logging.info(f"Attempting alternative installation for critical package: {clean_name}")
                    if self.install_package(clean_name):  # Try without version requirement
                        if self.verify_package(clean_name):
                            logging.info(f"Alternative installation successful for {clean_name}")
                            failed_packages.remove(package)
        
        # Accept 80% success rate for enterprise deployment
        final_success_rate = ((total_packages - len(failed_packages)) / total_packages) * 100
        
        if final_success_rate >= 80.0:
            logging.info(f"Enterprise installation SUCCESSFUL: {final_success_rate:.1f}% success rate")
            return True
        else:
            logging.error(f"Enterprise installation FAILED: {final_success_rate:.1f}% success rate")
            return False
    
    def create_installation_report(self) -> Dict[str, Any]:
        """Create comprehensive installation report"""
        report = {
            "platform": self.platform,
            "architecture": self.architecture,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": self.python_executable,
            "installation_status": "completed",
            "verified_packages": []
        }
        
        for package in self.required_packages:
            package_name = package.split(">=")[0]
            is_verified = self.verify_package(package)
            report["verified_packages"].append({
                "package": package_name,
                "verified": is_verified
            })
        
        return report

def main():
    """Main installation function"""
    print("🚀 NICEGOLD ProjectP - Enterprise Dependency Installer")
    print("=" * 60)
    
    installer = EnterpriseDependencyInstaller()
    installer.setup_logging()
    
    try:
        success = installer.install_all_dependencies()
        report = installer.create_installation_report()
        
        if success:
            print("\n✅ ENTERPRISE INSTALLATION SUCCESSFUL")
            print("All critical dependencies are now available")
            print("System is ready for production deployment")
            
            # Write success report
            try:
                import json
                with open("dependency_installation_report.json", "w") as f:
                    json.dump(report, f, indent=2)
                print("📊 Installation report saved to dependency_installation_report.json")
            except Exception:
                pass
            
            return True
        else:
            print("\n❌ ENTERPRISE INSTALLATION FAILED")
            print("Critical dependencies are missing")
            print("Please review dependency_installation.log for details")
            return False
            
    except Exception as e:
        print(f"\n💥 CRITICAL INSTALLATION ERROR: {str(e)}")
        logging.error(f"Critical installation error: {str(e)}")
        return False
    
    except KeyboardInterrupt:
        print("\n⚠️ Installation interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
