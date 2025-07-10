#!/usr/bin/env python3
"""
🧹 NICEGOLD Cleanup Script - Final Production Preparation
========================================================

This script cleans up all temporary fix/test/utility scripts after successful
production readiness validation. Only keeps essential production files.

KEEPS:
- ProjectP.py (main entry point)
- Core modules (core/, elliott_wave_modules/, menu_modules/)
- Configuration (config/, requirements.txt)
- Documentation (README.md, *.md files)

REMOVES:
- All fix_*.py scripts
- All test_*.py scripts  
- All validate_*.py scripts
- All demo_*.py scripts
- All utility scripts except ProjectP.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("🧹 NICEGOLD Production Cleanup")
    print("=" * 40)
    print("📅 Cleanup Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Define cleanup patterns
    cleanup_patterns = [
        "fix_*.py",
        "test_*.py", 
        "validate_*.py",
        "demo_*.py",
        "quick_*.py",
        "cleanup_*.py",
        "verify_*.py",
        "*_fixes.py",
        "*_fix.py"
    ]
    
    # Define files to explicitly keep (production essentials)
    keep_files = {
        "ProjectP.py",  # Main entry point
        "requirements.txt",
        "README.md",
        "install_all.py",
        "install_all.ps1", 
        "install_all.sh"
    }
    
    # Define files to explicitly remove (known temp files)
    remove_files = {
        "ProjectP_Advanced.py",
        "run_advanced.py", 
        "demo_advanced_logging.py",
        "fix_cuda_issues.py",
        "fix_elliott_cuda.py",
        "test_cuda_fix.py",
        "cleanup_cuda_files.py",
        "validate_single_entry_point.py",
        "fix_numpy_compatibility.py",
        "validate_production_readiness.py",
        "quick_numpy_fix.py"
    }
    
    removed_count = 0
    kept_count = 0
    
    print("\n🔍 Scanning for cleanup candidates...")
    
    # Get all Python files in root directory
    python_files = list(Path(".").glob("*.py"))
    
    for file_path in python_files:
        file_name = file_path.name
        
        # Always keep essential production files
        if file_name in keep_files:
            print(f"✅ KEEP: {file_name} (production essential)")
            kept_count += 1
            continue
        
        # Always remove known temporary files
        if file_name in remove_files:
            try:
                file_path.unlink()
                print(f"🗑️  REMOVED: {file_name} (temporary file)")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {file_name}: {e}")
            continue
        
        # Check against cleanup patterns
        should_remove = False
        for pattern in cleanup_patterns:
            if file_path.match(pattern):
                should_remove = True
                break
        
        if should_remove:
            try:
                file_path.unlink()
                print(f"🗑️  REMOVED: {file_name} (matches cleanup pattern)")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {file_name}: {e}")
        else:
            print(f"✅ KEEP: {file_name} (production file)")
            kept_count += 1
    
    # Clean up result files
    result_patterns = [
        "*_results_*.json",
        "*_report_*.json", 
        "*_validation_*.json"
    ]
    
    for pattern in result_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"🗑️  REMOVED: {file_path.name} (result file)")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path.name}: {e}")
    
    print("\n" + "=" * 40)
    print("📊 CLEANUP SUMMARY:")
    print(f"   🗑️  Files removed: {removed_count}")
    print(f"   ✅ Files kept: {kept_count}")
    print(f"   📁 Core modules: Preserved")
    print(f"   📄 Configuration: Preserved")
    print(f"   📚 Documentation: Preserved")
    
    print("\n🎉 Production cleanup completed!")
    print("🚀 System ready for production deployment")
    print("   Entry point: python ProjectP.py")
    
    # Show final directory structure
    print("\n📂 FINAL PRODUCTION STRUCTURE:")
    print("   ProjectP.py ← MAIN ENTRY POINT")
    print("   requirements.txt")
    print("   README.md")
    print("   core/ ← Core modules")
    print("   elliott_wave_modules/ ← ML modules")  
    print("   menu_modules/ ← Menu system")
    print("   config/ ← Configuration")
    print("   datacsv/ ← Data files")
    print("   logs/ ← Log files")
    print("   outputs/ ← Results")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
