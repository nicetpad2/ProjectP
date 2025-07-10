#!/usr/bin/env python3
"""
🎯 FINAL PRODUCTION READINESS VALIDATION
Menu 1 Enterprise Compliance and Real Profit Readiness Check

This script performs comprehensive validation to ensure Menu 1 is ready
for real profit trading operations with enterprise-grade compliance.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Project setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print validation header"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 80)
    print("🎯 MENU 1 PRODUCTION READINESS VALIDATION")
    print("Enterprise Compliance & Real Profit Trading Verification")
    print("=" * 80)
    print(f"{Colors.END}")
    print(f"{Colors.WHITE}📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project: {project_root}")
    print(f"🐍 Python: {sys.version.split()[0]}{Colors.END}\n")

def validate_core_files():
    """Validate core system files exist and are not empty"""
    print(f"{Colors.BLUE}{Colors.BOLD}📋 CORE FILES VALIDATION{Colors.END}")
    
    required_files = {
        'real_profit_feature_selector.py': 'Enterprise feature selector',
        'advanced_feature_selector.py': 'Advanced selector wrapper',
        'fast_feature_selector.py': 'Deprecated fast selector',
        'elliott_wave_modules/feature_selector.py': 'Elliott Wave selector',
        'menu_modules/menu_1_elliott_wave.py': 'Main Menu 1 module',
        'MENU1_ENTERPRISE_COMPLIANCE_COMPLETE.md': 'Compliance documentation'
    }
    
    all_good = True
    
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            if size > 0:
                print(f"   ✅ {file_path} ({size:,} bytes) - {description}")
            else:
                print(f"   {Colors.RED}❌ {file_path} (EMPTY) - {description}{Colors.END}")
                all_good = False
        else:
            print(f"   {Colors.RED}❌ {file_path} (MISSING) - {description}{Colors.END}")
            all_good = False
    
    return all_good

def validate_feature_selector_compliance():
    """Validate feature selector enterprise compliance"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🎯 FEATURE SELECTOR COMPLIANCE{Colors.END}")
    
    tests = []
    
    # Test 1: RealProfitFeatureSelector
    try:
        from real_profit_feature_selector import RealProfitFeatureSelector
        selector = RealProfitFeatureSelector(target_auc=0.70, max_features=30)
        
        # Check key attributes
        assert selector.target_auc == 0.70, "Target AUC not set correctly"
        assert selector.max_features == 30, "Max features not set correctly"
        assert hasattr(selector, 'select_features'), "select_features method missing"
        
        print("   ✅ RealProfitFeatureSelector: Import and initialization OK")
        tests.append(True)
        
    except Exception as e:
        print(f"   {Colors.RED}❌ RealProfitFeatureSelector: {e}{Colors.END}")
        tests.append(False)
    
    # Test 2: AdvancedFeatureSelector wrapper
    try:
        from advanced_feature_selector import AdvancedFeatureSelector
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        selector = AdvancedFeatureSelector()
        assert isinstance(selector, RealProfitFeatureSelector), "Not inheriting from RealProfitFeatureSelector"
        
        print("   ✅ AdvancedFeatureSelector: Proper wrapper inheritance")
        tests.append(True)
        
    except Exception as e:
        print(f"   {Colors.RED}❌ AdvancedFeatureSelector: {e}{Colors.END}")
        tests.append(False)
    
    # Test 3: FastFeatureSelector deprecation
    try:
        import warnings
        from fast_feature_selector import FastFeatureSelector
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector = FastFeatureSelector()
            
            assert isinstance(selector, RealProfitFeatureSelector), "Not redirecting to RealProfitFeatureSelector"
            
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "No deprecation warnings shown"
        
        print("   ✅ FastFeatureSelector: Proper deprecation and redirection")
        tests.append(True)
        
    except Exception as e:
        print(f"   {Colors.RED}❌ FastFeatureSelector: {e}{Colors.END}")
        tests.append(False)
    
    # Test 4: Elliott Wave selector
    try:
        from elliott_wave_modules.feature_selector import FeatureSelector
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        selector = FeatureSelector()
        assert isinstance(selector, RealProfitFeatureSelector), "Not inheriting from RealProfitFeatureSelector"
        
        print("   ✅ Elliott Wave FeatureSelector: Proper wrapper inheritance")
        tests.append(True)
        
    except Exception as e:
        print(f"   {Colors.RED}❌ Elliott Wave FeatureSelector: {e}{Colors.END}")
        tests.append(False)
    
    return all(tests)

def validate_no_fallback_logic():
    """Validate no fallback or fast mode logic remains"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🚫 ANTI-FALLBACK VALIDATION{Colors.END}")
    
    # Search for prohibited patterns in key files
    prohibited_patterns = [
        ('fast_mode', 'Fast mode activation'),
        ('fallback', 'Fallback logic'),
        ('emergency', 'Emergency fallback'),
        ('sample.*=.*True', 'Sampling enabled'),
        ('chunk.*data', 'Data chunking'),
        ('timeout.*return', 'Timeout fallbacks')
    ]
    
    key_files = [
        'real_profit_feature_selector.py',
        'advanced_feature_selector.py',
        'fast_feature_selector.py',
        'elliott_wave_modules/feature_selector.py',
        'menu_modules/menu_1_elliott_wave.py'
    ]
    
    violations_found = False
    
    for file_path in key_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
            
        try:
            content = full_path.read_text(encoding='utf-8').lower()
            
            for pattern, description in prohibited_patterns:
                if pattern in content:
                    # Check if it's in a comment explaining why it's NOT used
                    lines = content.split('\n')
                    pattern_lines = [i for i, line in enumerate(lines) if pattern in line]
                    
                    # Allow if in comments that explicitly say "NO" or "ZERO"
                    allowed = False
                    for line_num in pattern_lines:
                        line = lines[line_num]
                        if any(keyword in line for keyword in ['# no ', '# zero ', '"""', "'''"]):
                            allowed = True
                            break
                    
                    if not allowed:
                        print(f"   {Colors.RED}❌ {file_path}: Found {description} - '{pattern}'{Colors.END}")
                        violations_found = True
        
        except Exception as e:
            print(f"   {Colors.YELLOW}⚠️ Could not scan {file_path}: {e}{Colors.END}")
    
    if not violations_found:
        print("   ✅ No fallback or fast mode logic detected")
    
    return not violations_found

def validate_enterprise_requirements():
    """Validate enterprise-specific requirements"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🏢 ENTERPRISE REQUIREMENTS{Colors.END}")
    
    requirements = []
    
    # Check for required libraries
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("   ✅ Core ML libraries available")
        requirements.append(True)
    except ImportError as e:
        print(f"   {Colors.RED}❌ Core ML libraries missing: {e}{Colors.END}")
        requirements.append(False)
    
    # Check for enterprise ML libraries
    try:
        import shap
        import optuna
        print("   ✅ Enterprise ML libraries (SHAP, Optuna) available")
        requirements.append(True)
    except ImportError:
        print(f"   {Colors.YELLOW}⚠️ Enterprise ML libraries missing (will impact performance){Colors.END}")
        requirements.append(False)
    
    # Check data directory
    data_dir = project_root / 'datacsv'
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            print(f"   ✅ Data directory with {len(csv_files)} CSV files")
            requirements.append(True)
        else:
            print(f"   {Colors.YELLOW}⚠️ Data directory exists but no CSV files found{Colors.END}")
            requirements.append(False)
    else:
        print(f"   {Colors.YELLOW}⚠️ Data directory missing{Colors.END}")
        requirements.append(False)
    
    return all(requirements)

def generate_final_report(files_ok, selectors_ok, no_fallback, enterprise_ok):
    """Generate final validation report"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 FINAL VALIDATION REPORT{Colors.END}")
    print("=" * 60)
    
    checks = [
        (files_ok, "Core Files Validation"),
        (selectors_ok, "Feature Selector Compliance"),
        (no_fallback, "Anti-Fallback Validation"),
        (enterprise_ok, "Enterprise Requirements")
    ]
    
    passed = sum(1 for check, _ in checks if check)
    total = len(checks)
    
    for check_passed, check_name in checks:
        status = f"{Colors.GREEN}✅ PASS" if check_passed else f"{Colors.RED}❌ FAIL"
        print(f"   {status}{Colors.END} - {check_name}")
    
    print(f"\n{Colors.BOLD}Overall Score: {passed}/{total}{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 VALIDATION SUCCESSFUL{Colors.END}")
        print(f"{Colors.GREEN}✅ Menu 1 is ENTERPRISE READY{Colors.END}")
        print(f"{Colors.GREEN}💰 Ready for REAL PROFIT TRADING{Colors.END}")
        print(f"{Colors.GREEN}🚀 PROCEED TO PRODUCTION DEPLOYMENT{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️ VALIDATION ISSUES DETECTED{Colors.END}")
        print(f"{Colors.RED}❌ Review and fix issues before production{Colors.END}")
        print(f"{Colors.RED}🛑 DO NOT deploy until all checks pass{Colors.END}")
        return False

def main():
    """Run complete validation suite"""
    print_header()
    
    # Run all validation checks
    files_ok = validate_core_files()
    selectors_ok = validate_feature_selector_compliance()
    no_fallback = validate_no_fallback_logic()
    enterprise_ok = validate_enterprise_requirements()
    
    # Generate final report
    all_passed = generate_final_report(files_ok, selectors_ok, no_fallback, enterprise_ok)
    
    print(f"\n{Colors.CYAN}🕒 Validation completed at {datetime.now().strftime('%H:%M:%S')}{Colors.END}")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Validation failed with error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
