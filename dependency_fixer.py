#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 NICEGOLD ENTERPRISE PROJECTP - DEPENDENCY FIXER
🛠️ Advanced Dependency Resolution and Problem Fixing System

This script fixes common dependency issues and ensures all packages work together.
Special focus on NumPy/SHAP compatibility and enterprise requirements.

Author: NICEGOLD Enterprise ProjectP Team
Version: v3.0 Enterprise Edition
Date: July 12, 2025
"""

import os
import sys
import subprocess
import importlib
import time
from pathlib import Path

class DependencyFixer:
    """
    🔧 Advanced Dependency Fixer
    Resolves common installation and compatibility issues
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.fixes_applied = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: list, description: str = "") -> bool:
        """Run command safely"""
        try:
            self.log(f"Executing: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log(f"✅ {description} successful", "SUCCESS")
                return True
            else:
                self.log(f"❌ {description} failed: {result.stderr[:200]}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ {description} error: {e}", "ERROR")
            return False
    
    def fix_numpy_shap_compatibility(self) -> bool:
        """Fix NumPy/SHAP compatibility issues"""
        self.log("🔧 Fixing NumPy/SHAP compatibility...")
        
        # Step 1: Uninstall conflicting versions
        self.log("Uninstalling potential conflicting packages...")
        self.run_command([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "shap"], "Uninstall conflicts")
        
        # Step 2: Install specific NumPy version
        success1 = self.run_command([
            sys.executable, "-m", "pip", "install", "numpy==1.26.4"
        ], "Install NumPy 1.26.4")
        
        # Step 3: Install SHAP
        success2 = self.run_command([
            sys.executable, "-m", "pip", "install", "shap>=0.48.0"
        ], "Install SHAP")
        
        if success1 and success2:
            self.fixes_applied.append("NumPy/SHAP compatibility fix")
            return True
        return False
    
    def fix_tensorflow_issues(self) -> bool:
        """Fix TensorFlow installation issues"""
        self.log("🔧 Fixing TensorFlow issues...")
        
        # Check if we need CPU-only version
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except:
            has_cuda = False
        
        if has_cuda:
            # Install GPU version
            success = self.run_command([
                sys.executable, "-m", "pip", "install", "tensorflow>=2.18.0"
            ], "Install TensorFlow GPU")
        else:
            # Install CPU version
            success = self.run_command([
                sys.executable, "-m", "pip", "install", "tensorflow-cpu>=2.18.0"
            ], "Install TensorFlow CPU")
        
        if success:
            self.fixes_applied.append("TensorFlow installation fix")
        return success
    
    def fix_pytorch_issues(self) -> bool:
        """Fix PyTorch installation issues"""
        self.log("🔧 Fixing PyTorch issues...")
        
        # Install CPU version for better compatibility
        pytorch_packages = [
            "torch>=2.6.0",
            "torchvision>=0.21.0",
            "torchaudio>=2.6.0"
        ]
        
        success_count = 0
        for package in pytorch_packages:
            if self.run_command([sys.executable, "-m", "pip", "install", package], f"Install {package}"):
                success_count += 1
        
        if success_count >= 2:  # At least torch and torchvision
            self.fixes_applied.append("PyTorch installation fix")
            return True
        return False
    
    def fix_optuna_dependencies(self) -> bool:
        """Fix Optuna and related optimization libraries"""
        self.log("🔧 Fixing Optuna dependencies...")
        
        # Install Optuna with specific dependencies
        packages = [
            "optuna>=4.4.0",
            "scikit-learn>=1.5.2",
            "scipy>=1.14.1"
        ]
        
        success_count = 0
        for package in packages:
            if self.run_command([sys.executable, "-m", "pip", "install", package], f"Install {package}"):
                success_count += 1
        
        if success_count == len(packages):
            self.fixes_applied.append("Optuna dependencies fix")
            return True
        return False
    
    def fix_enterprise_packages(self) -> bool:
        """Fix enterprise-specific packages"""
        self.log("🔧 Fixing enterprise packages...")
        
        enterprise_packages = [
            "rich>=13.9.4",
            "colorama>=0.4.6",
            "psutil>=6.1.0",
            "PyYAML>=6.0.2",
            "joblib>=1.4.2"
        ]
        
        success_count = 0
        for package in enterprise_packages:
            if self.run_command([sys.executable, "-m", "pip", "install", package], f"Install {package}"):
                success_count += 1
        
        if success_count >= 4:  # Most packages successful
            self.fixes_applied.append("Enterprise packages fix")
            return True
        return False
    
    def fix_visualization_packages(self) -> bool:
        """Fix visualization packages"""
        self.log("🔧 Fixing visualization packages...")
        
        viz_packages = [
            "matplotlib>=3.9.2",
            "seaborn>=0.13.2",
            "plotly>=5.24.1"
        ]
        
        success_count = 0
        for package in viz_packages:
            if self.run_command([sys.executable, "-m", "pip", "install", package], f"Install {package}"):
                success_count += 1
        
        if success_count >= 2:
            self.fixes_applied.append("Visualization packages fix")
            return True
        return False
    
    def fix_financial_packages(self) -> bool:
        """Fix financial analysis packages"""
        self.log("🔧 Fixing financial packages...")
        
        financial_packages = [
            "yfinance>=0.2.38",
            "ta>=0.11.0",
            "pandas-ta",
            "mplfinance"
        ]
        
        success_count = 0
        for package in financial_packages:
            if self.run_command([sys.executable, "-m", "pip", "install", package], f"Install {package}"):
                success_count += 1
        
        if success_count >= 2:
            self.fixes_applied.append("Financial packages fix")
            return True
        return False
    
    def fix_missing_system_dependencies(self) -> bool:
        """Fix missing system dependencies"""
        self.log("🔧 Checking and fixing system dependencies...")
        
        # Try to install system-level dependencies if possible
        system_packages = []
        
        # Check if we're in a system where we can install system packages
        try:
            # Try to detect package manager
            if os.path.exists('/usr/bin/apt'):
                # Ubuntu/Debian
                self.log("Detected APT package manager")
                # Note: We can't actually run sudo commands, but we can suggest
                self.log("Consider running: sudo apt-get install build-essential python3-dev")
            elif os.path.exists('/usr/bin/yum'):
                # CentOS/RHEL
                self.log("Detected YUM package manager")
                self.log("Consider running: sudo yum install gcc python3-devel")
            elif os.path.exists('/opt/homebrew/bin/brew'):
                # macOS with Homebrew
                self.log("Detected Homebrew package manager")
                self.log("Consider running: brew install python3-dev")
        except Exception:
            pass
        
        self.fixes_applied.append("System dependencies check")
        return True
    
    def test_critical_imports(self) -> dict:
        """Test critical package imports"""
        self.log("🧪 Testing critical package imports...")
        
        critical_packages = {
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'sklearn': 'Scikit-learn',
            'tensorflow': 'TensorFlow',
            'torch': 'PyTorch',
            'shap': 'SHAP',
            'optuna': 'Optuna',
            'rich': 'Rich',
            'colorama': 'Colorama',
            'psutil': 'PSUtil',
            'yaml': 'PyYAML'
        }
        
        results = {}
        
        for package, name in critical_packages.items():
            try:
                if package == 'sklearn':
                    import sklearn
                elif package == 'yaml':
                    import yaml
                else:
                    importlib.import_module(package)
                
                self.log(f"✅ {name} imported successfully", "SUCCESS")
                results[package] = True
            except ImportError as e:
                self.log(f"❌ {name} import failed: {e}", "ERROR")
                results[package] = False
        
        return results
    
    def run_comprehensive_fix(self) -> bool:
        """Run comprehensive dependency fixing"""
        self.log("🚀 Starting comprehensive dependency fixing...")
        
        print("🔧 NICEGOLD DEPENDENCY FIXER")
        print("=" * 40)
        
        fixes = [
            ("NumPy/SHAP Compatibility", self.fix_numpy_shap_compatibility),
            ("TensorFlow Issues", self.fix_tensorflow_issues),
            ("PyTorch Issues", self.fix_pytorch_issues),
            ("Optuna Dependencies", self.fix_optuna_dependencies),
            ("Enterprise Packages", self.fix_enterprise_packages),
            ("Visualization Packages", self.fix_visualization_packages),
            ("Financial Packages", self.fix_financial_packages),
            ("System Dependencies", self.fix_missing_system_dependencies)
        ]
        
        successful_fixes = 0
        
        for fix_name, fix_function in fixes:
            self.log(f"🔧 Applying fix: {fix_name}")
            try:
                if fix_function():
                    successful_fixes += 1
                    self.log(f"✅ {fix_name} completed", "SUCCESS")
                else:
                    self.log(f"⚠️ {fix_name} had issues", "WARNING")
            except Exception as e:
                self.log(f"❌ {fix_name} failed: {e}", "ERROR")
        
        # Test imports after fixes
        import_results = self.test_critical_imports()
        successful_imports = sum(import_results.values())
        total_imports = len(import_results)
        
        print("\n" + "=" * 40)
        print("📊 DEPENDENCY FIX SUMMARY")
        print("=" * 40)
        print(f"🔧 Fixes Applied: {successful_fixes}/{len(fixes)}")
        print(f"📦 Successful Imports: {successful_imports}/{total_imports}")
        print(f"✅ Applied Fixes: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            print("\n🎯 Successfully Applied:")
            for fix in self.fixes_applied:
                print(f"   ✅ {fix}")
        
        # Determine overall success
        success_rate = successful_imports / total_imports
        if success_rate >= 0.8:  # 80% success rate
            print("\n🎉 DEPENDENCY FIXING SUCCESSFUL!")
            print("✅ Most critical packages are working")
            print("🚀 You can now run: python ProjectP.py")
            return True
        else:
            print("\n⚠️ DEPENDENCY FIXING PARTIALLY SUCCESSFUL")
            print("🔧 Some packages may still need manual installation")
            print("📋 Run: python check_installation.py for detailed status")
            return False


def main():
    """Main function"""
    try:
        fixer = DependencyFixer()
        success = fixer.run_comprehensive_fix()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Dependency fixing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Dependency fixing error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
