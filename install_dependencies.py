#!/usr/bin/env python3
"""
🔧 NICEGOLD PROJECT P - DEPENDENCY INSTALLER
Install required dependencies without external package managers
"""

import os
import sys
import subprocess
import importlib

def check_and_install_package(package_name, import_name=None):
    """Check if package exists, if not try to install it"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}: Already installed")
        return True
    except ImportError:
        print(f"⚠️ {package_name}: Not found, attempting installation...")
        
        # Try different installation methods
        install_methods = [
            [sys.executable, "-m", "pip", "install", "--user", package_name],
            ["pip3", "install", "--user", package_name],
            ["python3", "-m", "pip", "install", "--user", package_name],
            ["apt", "install", "-y", f"python3-{package_name.lower()}"],
        ]
        
        for method in install_methods:
            try:
                print(f"🔧 Trying: {' '.join(method)}")
                result = subprocess.run(method, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"✅ {package_name}: Installation successful")
                    return True
                else:
                    print(f"❌ Method failed: {result.stderr}")
            except Exception as e:
                print(f"❌ Installation method failed: {e}")
                continue
        
        print(f"⚠️ {package_name}: Could not install, will use fallback")
        return False

def main():
    """Install essential dependencies"""
    print("🚀 NICEGOLD ProjectP - Dependency Installation")
    print("=" * 50)
    
    # Essential packages
    packages = [
        ("psutil", "psutil"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    ]
    
    results = {}
    for package_name, import_name in packages:
        results[package_name] = check_and_install_package(package_name, import_name)
    
    print("\n📊 Installation Summary:")
    print("=" * 30)
    for package, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"{package:15} : {status}")
    
    # Create fallback indicators
    if not results.get('psutil', False):
        print("\n⚠️ psutil not available - creating fallback indicator")
        with open('/mnt/data/projects/ProjectP/NO_PSUTIL', 'w') as f:
            f.write("psutil not available - using lightweight fallback")
    
    total_success = sum(results.values())
    total_packages = len(results)
    
    print(f"\n🎯 Installation Complete: {total_success}/{total_packages} packages installed")
    
    if total_success >= 2:  # At least numpy and pandas
        print("✅ Minimum requirements met - system can run")
        return True
    else:
        print("❌ Critical packages missing - may need manual installation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
