#!/usr/bin/env python3
"""
🏢 ENTERPRISE COMPLIANCE VERIFICATION
Menu 1 Elliott Wave + SHAP + Optuna System

This script verifies that all enterprise requirements from 
MENU1_ELLWAVE_SHAP_OPTUNA_PLAN.md have been implemented correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports are available"""
    print("🔍 Testing Enterprise Component Imports...")
    
    try:
        # Core Menu 1 import
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Menu1ElliottWave imported successfully")
        
        # Enterprise Feature Selector
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        print("✅ EnterpriseShapOptunaFeatureSelector imported successfully")
        
        # Required libraries
        import shap, optuna
        print(f"✅ SHAP v{shap.__version__} available")
        print(f"✅ Optuna v{optuna.__version__} available")
        
        # Other Elliott Wave components
        from elliott_wave_modules import (
            ElliottWaveDataProcessor, CNNLSTMElliottWave, 
            DQNReinforcementAgent, ElliottWavePipelineOrchestrator
        )
        print("✅ All Elliott Wave modules imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_enterprise_compliance():
    """Test enterprise compliance features"""
    print("\n🏢 Testing Enterprise Compliance...")
    
    try:
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        
        # Initialize with enterprise parameters
        selector = EnterpriseShapOptunaFeatureSelector(
            target_auc=0.70,
            max_features=30
        )
        
        # Check enterprise parameters
        assert selector.target_auc == 0.70, "Target AUC not set correctly"
        assert selector.n_trials >= 100, "Insufficient Optuna trials for enterprise"
        assert selector.timeout >= 300, "Insufficient timeout for enterprise quality"
        
        print("✅ Enterprise parameters verified")
        print(f"   🎯 Target AUC: {selector.target_auc}")
        print(f"   ⚡ Optuna Trials: {selector.n_trials}")
        print(f"   ⏱️ Timeout: {selector.timeout}s")
        
        return True
    except Exception as e:
        print(f"❌ Enterprise compliance test failed: {str(e)}")
        return False

def test_no_fallbacks():
    """Verify no fallback methods exist"""
    print("\n🚫 Testing Zero Fallback Policy...")
    
    try:
        # Read the feature selector source
        feature_selector_path = Path("elliott_wave_modules/feature_selector.py")
        
        if not feature_selector_path.exists():
            print("❌ Feature selector file not found")
            return False
            
        with open(feature_selector_path, 'r') as f:
            content = f.read()
        
        # Check for forbidden patterns
        forbidden_patterns = [
            "_fallback",
            "mock",
            "dummy", 
            "simulation",
            "time.sleep",
            "simple_enhanced_pipeline"
        ]
        
        found_violations = []
        for pattern in forbidden_patterns:
            if pattern in content.lower():
                # Count occurrences
                count = content.lower().count(pattern)
                found_violations.append(f"{pattern} ({count} occurrences)")
        
        if found_violations:
            print("❌ Fallback violations found:")
            for violation in found_violations:
                print(f"   🚫 {violation}")
            return False
        else:
            print("✅ Zero fallback policy verified")
            print("   🚫 No forbidden patterns found")
            return True
            
    except Exception as e:
        print(f"❌ Fallback test failed: {str(e)}")
        return False

def test_data_files():
    """Test that real data files are available"""
    print("\n📊 Testing Real Data Availability...")
    
    try:
        data_dir = Path("datacsv")
        if not data_dir.exists():
            print("❌ datacsv directory not found")
            return False
        
        # Check for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV data files found")
            return False
        
        print(f"✅ Real data files available: {len(csv_files)} files")
        for csv_file in csv_files[:3]:  # Show first 3
            print(f"   📄 {csv_file.name}")
        
        return True
    except Exception as e:
        print(f"❌ Data file test failed: {str(e)}")
        return False

def test_menu_initialization():
    """Test Menu 1 can be initialized with enterprise components"""
    print("\n🎯 Testing Menu 1 Enterprise Initialization...")
    
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        
        # Initialize Menu 1
        menu = Menu1ElliottWave()
        
        # Check components
        assert hasattr(menu, 'feature_selector'), "Feature selector not initialized"
        assert hasattr(menu, 'data_processor'), "Data processor not initialized"
        assert hasattr(menu, 'cnn_lstm_engine'), "CNN-LSTM engine not initialized"
        assert hasattr(menu, 'dqn_agent'), "DQN agent not initialized"
        
        # Check enterprise feature selector type
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        assert isinstance(menu.feature_selector, EnterpriseShapOptunaFeatureSelector), \
            "Wrong feature selector type"
        
        print("✅ Menu 1 enterprise initialization successful")
        print(f"   🧠 Feature Selector: {type(menu.feature_selector).__name__}")
        print(f"   🎯 Target AUC: {menu.feature_selector.target_auc}")
        
        return True
    except Exception as e:
        print(f"❌ Menu initialization test failed: {str(e)}")
        return False

def main():
    """Run all enterprise compliance tests"""
    print("🏢 NICEGOLD ENTERPRISE MENU 1 COMPLIANCE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_imports),
        ("Enterprise Compliance", test_enterprise_compliance),
        ("Zero Fallback Policy", test_no_fallbacks),
        ("Real Data Files", test_data_files),
        ("Menu 1 Initialization", test_menu_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"🏆 ENTERPRISE COMPLIANCE RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("✅ ALL ENTERPRISE REQUIREMENTS SATISFIED")
        print("🚀 Menu 1 is PRODUCTION READY")
        return True
    else:
        print("❌ ENTERPRISE REQUIREMENTS NOT MET")
        print("🚫 Production deployment BLOCKED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
