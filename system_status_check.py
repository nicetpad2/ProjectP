#!/usr/bin/env python3
"""
📊 NICEGOLD ProjectP - Current Status Report
==========================================

This script provides a comprehensive status report of the NICEGOLD ProjectP system,
including environment health, dependency status, and production readiness.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

def get_system_info():
    """Get basic system information"""
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'cwd': os.getcwd(),
        'timestamp': datetime.now().isoformat()
    }

def check_files_exist():
    """Check if critical files exist"""
    critical_files = [
        'ProjectP.py',
        'requirements.txt',
        'README.md',
        'core/menu_system.py',
        'core/compliance.py',
        'elliott_wave_modules/feature_selector.py',
        'menu_modules/menu_1_elliott_wave.py'
    ]
    
    status = {}
    for file in critical_files:
        status[file] = Path(file).exists()
    
    return status

def check_data_files():
    """Check data files status"""
    data_files = [
        'datacsv/XAUUSD_M1.csv',
        'datacsv/XAUUSD_M15.csv'
    ]
    
    status = {}
    for file in data_files:
        path = Path(file)
        if path.exists():
            status[file] = {
                'exists': True,
                'size_mb': round(path.stat().st_size / (1024*1024), 2)
            }
        else:
            status[file] = {'exists': False}
    
    return status

def check_package_status():
    """Check package installation status"""
    packages = [
        'numpy', 'pandas', 'scikit-learn', 'shap', 'optuna',
        'tensorflow', 'torch', 'stable-baselines3', 'gymnasium'
    ]
    
    results = {}
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {package}; print({package}.__version__)'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results[package] = {
                    'status': 'OK',
                    'version': result.stdout.strip()
                }
            else:
                results[package] = {
                    'status': 'IMPORT_ERROR',
                    'error': result.stderr.strip()
                }
        except subprocess.TimeoutExpired:
            results[package] = {'status': 'TIMEOUT'}
        except Exception as e:
            results[package] = {'status': 'ERROR', 'error': str(e)}
    
    return results

def check_temp_files():
    """Check for temporary/fix files that should be cleaned up"""
    patterns = [
        'fix_*.py',
        'test_*.py',
        'validate_*.py',
        'verify_*.py',
        'quick_*.py',
        'cleanup_*.py',
        '*_fix*.py'
    ]
    
    temp_files = []
    for pattern in patterns:
        temp_files.extend(Path('.').glob(pattern))
    
    return [str(f) for f in temp_files]

def main():
    """Generate comprehensive status report"""
    print("📊 NICEGOLD ProjectP - System Status Report")
    print("=" * 50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System Information
    print("🖥️  SYSTEM INFORMATION")
    print("-" * 30)
    sys_info = get_system_info()
    print(f"Python: {sys_info['python_version'].split()[0]}")
    print(f"Platform: {sys_info['platform']}")
    print(f"Working Directory: {sys_info['cwd']}")
    print()
    
    # Critical Files Status
    print("📁 CRITICAL FILES STATUS")
    print("-" * 30)
    files_status = check_files_exist()
    for file, exists in files_status.items():
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{status}: {file}")
    
    missing_files = [f for f, exists in files_status.items() if not exists]
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} critical files missing!")
    else:
        print("\n✅ All critical files present")
    print()
    
    # Data Files Status
    print("📊 DATA FILES STATUS")
    print("-" * 30)
    data_status = check_data_files()
    for file, info in data_status.items():
        if info['exists']:
            print(f"✅ {file} ({info['size_mb']} MB)")
        else:
            print(f"❌ {file} (MISSING)")
    print()
    
    # Package Status
    print("📦 PACKAGE INSTALLATION STATUS")
    print("-" * 30)
    package_status = check_package_status()
    
    ok_packages = []
    error_packages = []
    
    for package, info in package_status.items():
        if info['status'] == 'OK':
            print(f"✅ {package}: {info['version']}")
            ok_packages.append(package)
        else:
            print(f"❌ {package}: {info['status']}")
            if 'error' in info:
                print(f"   Error: {info['error'][:100]}...")
            error_packages.append(package)
    
    print(f"\n📊 Package Summary: {len(ok_packages)}/{len(package_status)} working")
    print()
    
    # Temporary Files
    print("🧹 TEMPORARY FILES CHECK")
    print("-" * 30)
    temp_files = check_temp_files()
    if temp_files:
        print(f"⚠️  Found {len(temp_files)} temporary/fix files:")
        for file in temp_files[:10]:  # Show first 10
            print(f"   - {file}")
        if len(temp_files) > 10:
            print(f"   ... and {len(temp_files) - 10} more")
        print("\n💡 Consider running cleanup after successful validation")
    else:
        print("✅ No temporary files found - system is clean")
    print()
    
    # Overall Status Assessment
    print("🎯 OVERALL STATUS ASSESSMENT")
    print("-" * 30)
    
    critical_issues = []
    warnings = []
    
    # Check critical issues
    if missing_files:
        critical_issues.append(f"Missing {len(missing_files)} critical files")
    
    if len(error_packages) > 5:  # More than 5 packages failing
        critical_issues.append(f"{len(error_packages)} packages not working")
    
    if not data_status.get('datacsv/XAUUSD_M1.csv', {}).get('exists'):
        critical_issues.append("Primary data file missing")
    
    # Check warnings
    if temp_files:
        warnings.append(f"{len(temp_files)} temporary files need cleanup")
    
    if len(error_packages) > 0:
        warnings.append(f"{len(error_packages)} packages need attention")
    
    # Status determination
    if critical_issues:
        status = "🔴 CRITICAL ISSUES"
        print(f"{status}")
        for issue in critical_issues:
            print(f"   ❌ {issue}")
    elif warnings:
        status = "🟡 NEEDS ATTENTION"
        print(f"{status}")
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    else:
        status = "🟢 PRODUCTION READY"
        print(f"{status}")
        print("   ✅ All systems operational")
    
    print()
    
    # Next Steps
    print("🚀 RECOMMENDED NEXT STEPS")
    print("-" * 30)
    
    if critical_issues:
        print("1. Fix critical issues:")
        if missing_files:
            print("   - Restore missing critical files")
        if len(error_packages) > 5:
            print("   - Reinstall Python packages: pip install -r requirements.txt")
        if not data_status.get('datacsv/XAUUSD_M1.csv', {}).get('exists'):
            print("   - Ensure data files are in datacsv/ folder")
    elif warnings:
        print("1. Address warnings:")
        if error_packages:
            print("   - Fix package installation issues")
        if temp_files:
            print("   - Run cleanup script after validation")
        print("2. Test system: python ProjectP.py")
    else:
        print("1. System is ready! Run: python ProjectP.py")
        if temp_files:
            print("2. Optional: Clean up temporary files")
    
    print()
    print("📋 Status Report Complete")
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': status,
        'system_info': sys_info,
        'files_status': files_status,
        'data_status': data_status,
        'package_status': package_status,
        'temp_files': temp_files,
        'critical_issues': critical_issues,
        'warnings': warnings
    }
    
    report_file = f"system_status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"💾 Detailed report saved: {report_file}")
    
    return len(critical_issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
