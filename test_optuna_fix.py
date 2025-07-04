#!/usr/bin/env python3
"""
🧪 Test Optuna Fix - Feature Selector Bounds Issue
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project path
sys.path.append('/mnt/data/projects/ProjectP')

def test_fast_selector_bounds():
    """Test fast selector with various feature counts"""
    print("🧪 Testing Fast Feature Selector - Bounds Fix")
    print("=" * 50)
    
    try:
        from fast_feature_selector import FastEnterpriseFeatureSelector
        
        # Test with small number of features (edge case)
        print("📊 Test 1: Small dataset (5 features)")
        X_small = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feature_{i}' for i in range(5)])
        y_small = np.random.randint(0, 2, 1000)
        
        selector_small = FastEnterpriseFeatureSelector(max_features=20, target_auc=0.65)
        features_small, results_small = selector_small.select_features(X_small, y_small)
        
        print(f"✅ Small test: {len(features_small)} features selected")
        print(f"   AUC: {results_small.get('final_auc', 0):.4f}")
        
        # Test with medium dataset
        print("\n📊 Test 2: Medium dataset (15 features)")
        X_medium = pd.DataFrame(np.random.randn(2000, 15), columns=[f'feature_{i}' for i in range(15)])
        y_medium = np.random.randint(0, 2, 2000)
        
        selector_medium = FastEnterpriseFeatureSelector(max_features=10, target_auc=0.65)
        features_medium, results_medium = selector_medium.select_features(X_medium, y_medium)
        
        print(f"✅ Medium test: {len(features_medium)} features selected")
        print(f"   AUC: {results_medium.get('final_auc', 0):.4f}")
        
        # Test with normal dataset
        print("\n📊 Test 3: Normal dataset (30 features)")
        X_normal = pd.DataFrame(np.random.randn(3000, 30), columns=[f'feature_{i}' for i in range(30)])
        y_normal = np.random.randint(0, 2, 3000)
        
        selector_normal = FastEnterpriseFeatureSelector(max_features=20, target_auc=0.65)
        features_normal, results_normal = selector_normal.select_features(X_normal, y_normal)
        
        print(f"✅ Normal test: {len(features_normal)} features selected")
        print(f"   AUC: {results_normal.get('final_auc', 0):.4f}")
        
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        print(f"  Small (5 features): {len(features_small)} selected")
        print(f"  Medium (15 features): {len(features_medium)} selected") 
        print(f"  Normal (30 features): {len(features_normal)} selected")
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_selector_params():
    """Test advanced selector with proper parameters"""
    print("\n🧪 Testing Advanced Feature Selector - Parameters")
    print("=" * 50)
    
    try:
        from advanced_feature_selector import AdvancedEnterpriseFeatureSelector
        
        # Test with correct parameters
        X = pd.DataFrame(np.random.randn(1500, 20), columns=[f'feature_{i}' for i in range(20)])
        y = np.random.randint(0, 2, 1500)
        
        selector = AdvancedEnterpriseFeatureSelector(
            target_auc=0.65,
            max_features=15,
            fast_mode=True,
            auto_fast_mode=True,
            large_dataset_threshold=2000
        )
        
        features, results = selector.select_features(X, y)
        
        print(f"✅ Advanced selector: {len(features)} features selected")
        print(f"   AUC: {results.get('best_auc', 0):.4f}")
        print(f"   Target achieved: {results.get('target_achieved', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 OPTUNA BOUNDS FIX TESTING")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Fast selector bounds
    if test_fast_selector_bounds():
        success_count += 1
    
    # Test 2: Advanced selector parameters
    if test_advanced_selector_params():
        success_count += 1
    
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    print(f"  Passed: {success_count}/{total_tests}")
    print(f"  Success Rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Bounds fix successful!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
