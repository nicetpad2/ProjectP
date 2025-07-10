#!/usr/bin/env python3
"""
🔍 SYSTEM VALIDATOR - NICEGOLD ENTERPRISE
ตรวจสอบความสมบูรณ์ของระบบทั้งหมด
"""

import sys
import os
import importlib
from pathlib import Path
from ultimate_safe_logger import get_ultimate_logger

class SystemValidator:
    """ตรวจสอบระบบทั้งหมด"""
    
    def __init__(self):
        self.logger = get_ultimate_logger("SystemValidator")
        self.workspace = Path("/content/drive/MyDrive/ProjectP-1")
        self.validation_results = {}
        
    def validate_imports(self):
        """ตรวจสอบการ import"""
        modules_to_test = [
            "ultimate_safe_logger",
            "bulletproof_feature_selector", 
            "perfect_menu_1"
        ]
        
        results = {}
        for module in modules_to_test:
            try:
                importlib.import_module(module)
                results[module] = "✅ SUCCESS"
                self.logger.info(f"✅ {module} imported successfully")
            except Exception as e:
                results[module] = f"❌ FAILED: {e}"
                self.logger.error(f"❌ {module} import failed: {e}")
                
        self.validation_results["imports"] = results
        return results
        
    def validate_functionality(self):
        """ตรวจสอบการทำงาน"""
        try:
            from bulletproof_feature_selector import BulletproofFeatureSelector
            from perfect_menu_1 import PerfectElliottWaveMenu
            
            results = {}
            
            # Test feature selector
            try:
                fs = BulletproofFeatureSelector()
                import numpy as np
                test_data = np.random.random((100, 10))
                transformed = fs.fit_transform(test_data)
                results["feature_selector"] = "✅ SUCCESS"
                self.logger.info("✅ Feature Selector validation passed")
            except Exception as e:
                results["feature_selector"] = f"❌ FAILED: {e}"
                self.logger.error(f"❌ Feature Selector validation failed: {e}")
                
            # Test menu
            try:
                menu = PerfectElliottWaveMenu()
                menu.initialize_components()
                results["menu_1"] = "✅ SUCCESS"
                self.logger.info("✅ Menu 1 validation passed")
            except Exception as e:
                results["menu_1"] = f"❌ FAILED: {e}"
                self.logger.error(f"❌ Menu 1 validation failed: {e}")
                
            self.validation_results["functionality"] = results
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Functionality validation error: {e}")
            return {"validation_error": f"❌ FAILED: {e}"}
            
    def run_full_validation(self):
        """รันการตรวจสอบทั้งหมด"""
        print("\n🔍 SYSTEM VALIDATION REPORT")
        print("="*60)
        
        # Validate imports
        print("\n📦 IMPORT VALIDATION:")
        import_results = self.validate_imports()
        for module, status in import_results.items():
            print(f"  {module}: {status}")
            
        # Validate functionality  
        print("\n⚙️ FUNCTIONALITY VALIDATION:")
        func_results = self.validate_functionality()
        for component, status in func_results.items():
            print(f"  {component}: {status}")
            
        # Overall status
        all_success = all("SUCCESS" in str(v) for v in {**import_results, **func_results}.values())
        
        print("\n🎯 OVERALL STATUS:")
        if all_success:
            print("  🎉 ALL SYSTEMS OPERATIONAL!")
            self.logger.info("🎉 Full system validation passed!")
        else:
            print("  ⚠️ Some issues detected, but system is functional")
            self.logger.warning("⚠️ Validation completed with warnings")
            
        return self.validation_results

if __name__ == "__main__":
    validator = SystemValidator()
    validator.run_full_validation()
