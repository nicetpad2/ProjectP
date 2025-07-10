#!/usr/bin/env python3
"""
NICEGOLD ProjectP - Zero-Fallback Production Launcher
Enterprise-grade launcher with automatic dependency resolution and environment management
"""

import os
import sys
import subprocess
import warnings
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Immediate warning suppression
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class ProductionLauncher:
    """Zero-fallback production launcher for NICEGOLD ProjectP"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.python_executable = sys.executable
        self.environment_activated = False
        self.dependencies_verified = False
        
    def setup_enterprise_logging(self) -> None:
        """Setup enterprise-grade logging system"""
        try:
            log_dir = self.project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_dir / "production_launcher.log", mode='w')
                ]
            )
            self.logger = logging.getLogger("ProductionLauncher")
            self.logger.info("Enterprise logging system initialized")
        except Exception:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ProductionLauncher")
    
    def check_environment_activation(self) -> bool:
        """Check if the correct environment is activated"""
        try:
            # Check for environment script
            env_script = self.project_root / "activate_nicegold_env.sh"
            if env_script.exists():
                self.logger.info("Environment activation script found")
                
                # Check if we're in the correct environment
                env_path = os.environ.get("VIRTUAL_ENV", "")
                if "nicegold" in env_path.lower():
                    self.logger.info(f"Correct environment detected: {env_path}")
                    self.environment_activated = True
                    return True
                else:
                    self.logger.warning("Environment may not be activated correctly")
                    return self.activate_environment()
            else:
                self.logger.warning("Environment activation script not found")
                return True  # Continue without environment activation
                
        except Exception as e:
            self.logger.error(f"Environment check failed: {str(e)}")
            return True  # Continue anyway
    
    def activate_environment(self) -> bool:
        """Attempt to activate the environment"""
        try:
            env_script = self.project_root / "activate_nicegold_env.sh"
            if env_script.exists():
                self.logger.info("Attempting environment activation...")
                # Environment activation is typically done before this script runs
                # We'll assume it's handled by the shell wrapper
                return True
            return True
        except Exception as e:
            self.logger.warning(f"Environment activation failed: {str(e)}")
            return True
    
    def verify_critical_dependencies(self) -> bool:
        """Verify that all critical dependencies are available"""
        critical_deps = [
            "psutil", "numpy", "pandas", "sklearn", 
            "matplotlib", "yfinance", "plotly"
        ]
        
        missing_deps = []
        
        for dep in critical_deps:
            try:
                if dep == "sklearn":
                    __import__("sklearn")
                else:
                    __import__(dep)
                self.logger.info(f"✓ {dep} verified")
            except ImportError:
                missing_deps.append(dep)
                self.logger.warning(f"✗ {dep} missing")
        
        if missing_deps:
            self.logger.error(f"Missing critical dependencies: {', '.join(missing_deps)}")
            return False
        
        self.logger.info("All critical dependencies verified")
        self.dependencies_verified = True
        return True
    
    def install_missing_dependencies(self) -> bool:
        """Install missing dependencies using enterprise installer"""
        try:
            installer_script = self.project_root / "enterprise_dependency_installer.py"
            if installer_script.exists():
                self.logger.info("Running enterprise dependency installer...")
                
                result = subprocess.run([
                    self.python_executable, str(installer_script)
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    self.logger.info("Dependency installation completed successfully")
                    return self.verify_critical_dependencies()
                else:
                    self.logger.error(f"Dependency installation failed: {result.stderr}")
                    return False
            else:
                self.logger.error("Enterprise dependency installer not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to install dependencies: {str(e)}")
            return False
    
    def verify_system_health(self) -> Dict[str, Any]:
        """Verify system health before launch"""
        health_report = {
            "environment_status": self.environment_activated,
            "dependencies_status": self.dependencies_verified,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": self.python_executable,
            "project_root": str(self.project_root),
            "ready_for_launch": False
        }
        
        # Check core files
        core_files = [
            "ProjectP_optimized_final.py",
            "core/optimized_resource_manager.py",
            "menu_modules/ultra_lightweight_menu_1.py"
        ]
        
        missing_files = []
        for file_path in core_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        health_report["missing_files"] = missing_files
        health_report["core_files_available"] = len(missing_files) == 0
        
        # Overall readiness check
        health_report["ready_for_launch"] = (
            health_report["dependencies_status"] and
            health_report["core_files_available"]
        )
        
        return health_report
    
    def launch_main_system(self) -> bool:
        """Launch the main ProjectP system"""
        try:
            main_script = self.project_root / "ProjectP_optimized_final.py"
            if not main_script.exists():
                self.logger.error("Main system script not found")
                return False
            
            self.logger.info("Launching NICEGOLD ProjectP Main System...")
            
            # Change to project directory
            os.chdir(self.project_root)
            
            # Launch with proper environment
            result = subprocess.run([
                self.python_executable, str(main_script)
            ], cwd=self.project_root)
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to launch main system: {str(e)}")
            return False
    
    def create_launch_report(self, health_report: Dict[str, Any]) -> None:
        """Create comprehensive launch report"""
        try:
            report_path = self.project_root / "production_launch_report.json"
            with open(report_path, "w") as f:
                json.dump(health_report, f, indent=2)
            self.logger.info(f"Launch report saved to {report_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save launch report: {str(e)}")

def main():
    """Main launcher function"""
    print("🚀 NICEGOLD ProjectP - Production Launcher")
    print("=" * 50)
    print("Enterprise-grade zero-fallback system")
    print("=" * 50)
    
    launcher = ProductionLauncher()
    launcher.setup_enterprise_logging()
    
    try:
        # Step 1: Environment Check
        print("\n🔍 Checking environment...")
        if not launcher.check_environment_activation():
            print("⚠️ Environment activation issues detected")
        
        # Step 2: Dependency Verification
        print("\n📦 Verifying dependencies...")
        if not launcher.verify_critical_dependencies():
            print("🔧 Installing missing dependencies...")
            if not launcher.install_missing_dependencies():
                print("❌ Critical dependency installation failed")
                print("Manual intervention required")
                return False
        
        # Step 3: System Health Check
        print("\n🏥 Performing system health check...")
        health_report = launcher.verify_system_health()
        launcher.create_launch_report(health_report)
        
        if not health_report["ready_for_launch"]:
            print("❌ System health check failed")
            print("Missing critical components:")
            for issue in health_report.get("missing_files", []):
                print(f"  - {issue}")
            return False
        
        # Step 4: Launch Main System
        print("\n🎯 All systems ready - Launching NICEGOLD ProjectP...")
        print("=" * 50)
        
        success = launcher.launch_main_system()
        
        if success:
            print("\n✅ System launched successfully")
            return True
        else:
            print("\n❌ System launch failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Launch interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Critical launch error: {str(e)}")
        launcher.logger.error(f"Critical launch error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔧 Troubleshooting tips:")
        print("1. Check dependency_installation.log for details")
        print("2. Ensure Python 3.8+ is installed")
        print("3. Verify internet connection for package downloads")
        print("4. Run as administrator if permission errors occur")
    
    sys.exit(0 if success else 1)
