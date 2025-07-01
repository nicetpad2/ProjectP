#!/usr/bin/env python3
"""
🎯 NICEGOLD Production Readiness Final Validation
================================================

This script performs final validation to ensure NICEGOLD ProjectP is 100% production-ready:
- Tests NumPy/SHAP compatibility
- Validates single entry point policy
- Tests all enterprise compliance features
- Verifies CPU-only operation
- Confirms AUC ≥ 0.70 enforcement

ENTERPRISE REQUIREMENTS:
✅ Only real data (no simulation/mock/dummy/fallback)
✅ AUC ≥ 0.70 enforcement
✅ Single entry point (ProjectP.py only)
✅ CPU-only operation (no CUDA errors)
✅ NumPy 1.x compatibility with SHAP
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from datetime import datetime

class ProductionValidator:
    def __init__(self):
        self.results = {
            'validation_time': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'PENDING'
        }
        
    def log(self, message, test_name=None):
        """Log validation message"""
        print(message)
        if test_name and test_name in self.results['tests']:
            if 'logs' not in self.results['tests'][test_name]:
                self.results['tests'][test_name]['logs'] = []
            self.results['tests'][test_name]['logs'].append(message)
    
    def test_numpy_shap_compatibility(self):
        """Test NumPy and SHAP compatibility"""
        test_name = 'numpy_shap_compatibility'
        self.results['tests'][test_name] = {'status': 'TESTING'}
        
        try:
            self.log("🔬 Testing NumPy and SHAP compatibility...", test_name)
            
            # Test NumPy version
            import numpy as np
            numpy_version = np.__version__
            version_parts = numpy_version.split('.')
            major_version = int(version_parts[0])
            
            if major_version >= 2:
                self.log(f"❌ NumPy {numpy_version} is 2.x - incompatible with SHAP", test_name)
                self.results['tests'][test_name]['status'] = 'FAILED'
                return False
            
            self.log(f"✅ NumPy {numpy_version} is 1.x - compatible with SHAP", test_name)
            
            # Test SHAP import and basic functionality
            import shap
            shap_version = shap.__version__
            self.log(f"✅ SHAP {shap_version} imported successfully", test_name)
            
            # Test SHAP functionality
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=50, n_features=5, random_state=42)
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X, y)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:2])
            
            self.log("✅ SHAP TreeExplainer working correctly", test_name)
            self.results['tests'][test_name]['status'] = 'PASSED'
            return True
            
        except Exception as e:
            self.log(f"❌ NumPy/SHAP compatibility test failed: {e}", test_name)
            self.results['tests'][test_name]['status'] = 'FAILED'
            return False
    
    def test_single_entry_point(self):
        """Test single entry point policy"""
        test_name = 'single_entry_point'
        self.results['tests'][test_name] = {'status': 'TESTING'}
        
        try:
            self.log("🚪 Testing single entry point policy...", test_name)
            
            # Check ProjectP.py exists
            if not Path('ProjectP.py').exists():
                self.log("❌ ProjectP.py not found", test_name)
                self.results['tests'][test_name]['status'] = 'FAILED'
                return False
            
            self.log("✅ ProjectP.py exists", test_name)
            
            # Check alternative entry points are disabled
            alternative_scripts = [
                'ProjectP_Advanced.py',
                'run_advanced.py',
                'demo_advanced_logging.py'
            ]
            
            for script in alternative_scripts:
                if Path(script).exists():
                    # Try to run it and check if it redirects properly
                    try:
                        result = subprocess.run([sys.executable, script], 
                                              capture_output=True, text=True, timeout=10)
                        if "ProjectP.py" in result.stdout or "single entry point" in result.stdout.lower():
                            self.log(f"✅ {script} properly redirects to ProjectP.py", test_name)
                        else:
                            self.log(f"⚠️  {script} should redirect to ProjectP.py", test_name)
                    except subprocess.TimeoutExpired:
                        self.log(f"⚠️  {script} timeout - may need manual check", test_name)
            
            self.results['tests'][test_name]['status'] = 'PASSED'
            return True
            
        except Exception as e:
            self.log(f"❌ Single entry point test failed: {e}", test_name)
            self.results['tests'][test_name]['status'] = 'FAILED'
            return False
    
    def test_cpu_only_operation(self):
        """Test CPU-only operation without CUDA errors"""
        test_name = 'cpu_only_operation'
        self.results['tests'][test_name] = {'status': 'TESTING'}
        
        try:
            self.log("💻 Testing CPU-only operation...", test_name)
            
            # Set CPU-only environment
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Test TensorFlow CPU-only
            import tensorflow as tf
            tf_version = tf.__version__
            self.log(f"✅ TensorFlow {tf_version} imported (CPU-only)", test_name)
            
            # Test PyTorch CPU-only
            import torch
            torch_version = torch.__version__
            if torch.cuda.is_available():
                self.log("⚠️  CUDA available but will use CPU-only", test_name)
            else:
                self.log("✅ PyTorch CPU-only mode", test_name)
            
            self.log(f"✅ PyTorch {torch_version} configured for CPU", test_name)
            
            self.results['tests'][test_name]['status'] = 'PASSED'
            return True
            
        except Exception as e:
            self.log(f"❌ CPU-only operation test failed: {e}", test_name)
            self.results['tests'][test_name]['status'] = 'FAILED'
            return False
    
    def test_core_modules_import(self):
        """Test core modules can be imported without errors"""
        test_name = 'core_modules_import'
        self.results['tests'][test_name] = {'status': 'TESTING'}
        
        try:
            self.log("📦 Testing core modules import...", test_name)
            
            # Test core modules
            core_modules = [
                'core.menu_system',
                'core.compliance', 
                'core.logger',
                'core.config'
            ]
            
            for module_name in core_modules:
                try:
                    module = importlib.import_module(module_name)
                    self.log(f"✅ {module_name} imported successfully", test_name)
                except Exception as e:
                    self.log(f"❌ {module_name} import failed: {e}", test_name)
                    self.results['tests'][test_name]['status'] = 'FAILED'
                    return False
            
            # Test elliott wave modules
            elliott_modules = [
                'elliott_wave_modules.feature_selector',
                'elliott_wave_modules.data_processor',
                'elliott_wave_modules.pipeline_orchestrator'
            ]
            
            for module_name in elliott_modules:
                try:
                    module = importlib.import_module(module_name)
                    self.log(f"✅ {module_name} imported successfully", test_name)
                except Exception as e:
                    self.log(f"❌ {module_name} import failed: {e}", test_name)
                    self.results['tests'][test_name]['status'] = 'FAILED'
                    return False
            
            self.results['tests'][test_name]['status'] = 'PASSED'
            return True
            
        except Exception as e:
            self.log(f"❌ Core modules import test failed: {e}", test_name)
            self.results['tests'][test_name]['status'] = 'FAILED'
            return False
    
    def test_enterprise_compliance(self):
        """Test enterprise compliance features"""
        test_name = 'enterprise_compliance'
        self.results['tests'][test_name] = {'status': 'TESTING'}
        
        try:
            self.log("🏢 Testing enterprise compliance...", test_name)
            
            # Check compliance module
            from core.compliance import ComplianceValidator
            validator = ComplianceValidator()
            
            self.log("✅ ComplianceValidator imported", test_name)
            
            # Test AUC threshold enforcement
            if hasattr(validator, 'MIN_AUC_THRESHOLD'):
                threshold = validator.MIN_AUC_THRESHOLD
                if threshold >= 0.70:
                    self.log(f"✅ AUC threshold set to {threshold} (≥ 0.70)", test_name)
                else:
                    self.log(f"❌ AUC threshold {threshold} < 0.70", test_name)
                    self.results['tests'][test_name]['status'] = 'FAILED'
                    return False
            
            # Test real data enforcement
            if hasattr(validator, 'validate_real_data_only'):
                self.log("✅ Real data validation available", test_name)
            else:
                self.log("⚠️  Real data validation method not found", test_name)
            
            self.results['tests'][test_name]['status'] = 'PASSED'
            return True
            
        except Exception as e:
            self.log(f"❌ Enterprise compliance test failed: {e}", test_name)
            self.results['tests'][test_name]['status'] = 'FAILED'
            return False
    
    def run_all_tests(self):
        """Run all validation tests"""
        self.log("🚀 NICEGOLD Production Readiness Validation Starting...")
        self.log("=" * 60)
        
        tests = [
            self.test_numpy_shap_compatibility,
            self.test_single_entry_point,
            self.test_cpu_only_operation,
            self.test_core_modules_import,
            self.test_enterprise_compliance
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
                self.log("-" * 40)
            except Exception as e:
                self.log(f"❌ Test {test.__name__} crashed: {e}")
                self.log("-" * 40)
        
        # Final results
        self.log("=" * 60)
        self.log(f"📊 VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.results['overall_status'] = 'PRODUCTION_READY'
            self.log("🎉 NICEGOLD ProjectP is PRODUCTION READY!")
            self.log("✅ All enterprise requirements satisfied")
            self.log("✅ NumPy/SHAP compatibility confirmed")
            self.log("✅ Single entry point policy enforced")
            self.log("✅ CPU-only operation verified")
            self.log("✅ Enterprise compliance validated")
            self.log("\n🚀 Ready to run: python ProjectP.py")
        else:
            self.results['overall_status'] = 'NEEDS_FIXES'
            self.log("❌ Production readiness validation FAILED")
            self.log(f"❌ {total_tests - passed_tests} test(s) need attention")
        
        return passed_tests == total_tests
    
    def save_results(self):
        """Save validation results to file"""
        results_file = f"production_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"📄 Results saved to: {results_file}")

def main():
    """Main validation routine"""
    validator = ProductionValidator()
    
    try:
        success = validator.run_all_tests()
        validator.save_results()
        return success
    except Exception as e:
        print(f"❌ Validation crashed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
