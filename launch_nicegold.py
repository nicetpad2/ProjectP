#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - PRODUCTION LAUNCHER
=========================================================

🎯 Features:
- 🛡️ Enterprise-grade error handling
- 🚀 Automatic environment detection and activation
- ⚡ Zero-fallback policy implementation
- 📊 Complete system optimization
- 🔧 Comprehensive dependency management

⚠️ NO FALLBACKS ALLOWED - ENTERPRISE PRODUCTION ONLY
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_environment():
    """Check if we're in the correct environment"""
    python_path = sys.executable
    required_path = "/home/ACER/.cache/nicegold_env"
    
    if required_path in python_path:
        print("✅ Correct environment detected")
        return True
    else:
        print(f"❌ Wrong environment: {python_path}")
        print(f"✅ Required: {required_path}")
        return False

def activate_environment():
    """Activate the NICEGOLD environment"""
    activation_script = Path("activate_nicegold_env.sh")
    env_path = Path("/home/ACER/.cache/nicegold_env/nicegold_enterprise_env/bin/activate")
    
    if activation_script.exists():
        print("🔧 Activating environment via script...")
        try:
            # Use the activation script with proper python execution
            cmd = f"bash activate_nicegold_env.sh && python3 -c 'import sys; print(sys.executable)'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0 and "nicegold_env" in result.stdout:
                print("✅ Environment activation successful")
                return True
            else:
                print(f"❌ Activation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Activation error: {e}")
            return False
    
    elif env_path.exists():
        print("🔧 Activating environment directly...")
        try:
            # Direct activation
            cmd = f"source {env_path} && python3 -c 'import sys; print(sys.executable)'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0 and "nicegold_env" in result.stdout:
                print("✅ Direct environment activation successful")
                return True
            else:
                print(f"❌ Direct activation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Direct activation error: {e}")
            return False
    
    else:
        print("❌ No activation method available")
        return False

def check_dependencies():
    """Check critical dependencies"""
    required_modules = [
        'psutil', 'numpy', 'pandas', 'sklearn', 'tensorflow', 
        'torch', 'rich', 'pathlib', 'datetime'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"❌ {module}")
    
    if missing:
        print(f"\n🚨 Missing modules: {missing}")
        return False
    else:
        print("\n✅ All dependencies satisfied")
        return True

def launch_optimized_system():
    """Launch the optimized NICEGOLD system"""
    try:
        # Import the optimized system
        from ProjectP_optimized_final import main as optimized_main
        print("🚀 Launching optimized system...")
        optimized_main()
        return True
    except Exception as e:
        print(f"❌ Optimized system launch failed: {e}")
        return False

def launch_in_environment():
    """Launch the system within the proper environment"""
    activation_script = Path("activate_nicegold_env.sh")
    
    if activation_script.exists():
        print("🚀 Launching with environment activation...")
        try:
            # Execute within activated environment
            cmd = "bash activate_nicegold_env.sh && python3 ProjectP_optimized_final.py"
            result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Environment launch error: {e}")
            return False
    else:
        print("❌ No activation script found")
        return False

def main():
    """Main launcher function"""
    print("🏢 NICEGOLD ENTERPRISE PROJECTP - PRODUCTION LAUNCHER")
    print("=" * 80)
    
    # Step 1: Check if we're in the right environment
    if not check_environment():
        print("\n🔧 Environment activation required...")
        
        # Try to activate and relaunch
        if launch_in_environment():
            print("✅ System launched successfully in environment")
            return
        else:
            print("❌ Failed to launch in environment")
            print("\n📋 Manual activation required:")
            print("1. cd /mnt/data/projects/ProjectP")
            print("2. ./activate_nicegold_env.sh")
            print("3. python3 ProjectP_optimized_final.py")
            return
    
    # Step 2: Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        print("🔧 Please ensure environment is properly activated")
        return
    
    # Step 3: Launch optimized system
    print("\n🚀 All checks passed, launching optimized system...")
    if not launch_optimized_system():
        print("❌ System launch failed")
        return
    
    print("✅ NICEGOLD Enterprise system completed successfully")

if __name__ == "__main__":
    main()
