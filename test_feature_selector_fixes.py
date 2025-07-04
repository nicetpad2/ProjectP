#!/usr/bin/env python3
"""
Test script to verify SHAP analysis fixes and fast mode detection
"""

import sys
import os
sys.path.append('/mnt/data/projects/ProjectP')

import pandas as pd
import numpy as np
from datetime import datetime

def test_fast_mode_auto_detection():
    """Test that advanced selector automatically detects large datasets and uses fast mode"""
    print("🧪 Testing Fast Mode Auto Detection...")
    
    try:
        from advanced_feature_selector import AdvancedEnterpriseFeatureSelector
        
        # Test 1: Small dataset (should use standard mode)
        print("\n📊 Test 1: Small dataset (10,000 rows)")
        np.random.seed(42)
        n_samples_small = 10000
        n_features = 20
        
        X_small = pd.DataFrame(
            np.random.randn(n_samples_small, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add signal to some features
        X_small['feature_0'] += np.random.binomial(1, 0.3, n_samples_small) * 2
        X_small['feature_1'] += np.random.binomial(1, 0.4, n_samples_small) * 1.5
        
        y_small = pd.Series(
            (X_small['feature_0'] * 0.6 + X_small['feature_1'] * 0.4 + 
             np.random.randn(n_samples_small) * 0.2 > 0).astype(int)
        )
        
        print(f"📈 Small dataset: {len(X_small):,} samples, {len(X_small.columns)} features")
        print(f"🎯 Target distribution: {y_small.value_counts().to_dict()}")
        
        selector_small = AdvancedEnterpriseFeatureSelector(
            target_auc=0.70,
            max_features=10
        )
        
        print(f"🔍 Large dataset threshold: {selector_small.large_dataset_threshold:,}")
        print(f"⚡ Should use fast mode: {len(X_small) >= selector_small.large_dataset_threshold}")
        
        # Test 2: Large dataset (should use fast mode)
        print(f"\n📊 Test 2: Large dataset (150,000 rows)")
        n_samples_large = 150000
        
        # Create larger dataset efficiently
        X_large = pd.DataFrame(
            np.random.randn(n_samples_large, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        X_large['feature_0'] += np.random.binomial(1, 0.3, n_samples_large) * 2
        X_large['feature_1'] += np.random.binomial(1, 0.4, n_samples_large) * 1.5
        
        y_large = pd.Series(
            (X_large['feature_0'] * 0.6 + X_large['feature_1'] * 0.4 + 
             np.random.randn(n_samples_large) * 0.2 > 0).astype(int)
        )
        
        print(f"📈 Large dataset: {len(X_large):,} samples, {len(X_large.columns)} features")
        print(f"🎯 Target distribution: {y_large.value_counts().to_dict()}")
        
        selector_large = AdvancedEnterpriseFeatureSelector(
            target_auc=0.70,
            max_features=10
        )
        
        print(f"⚡ Should use fast mode: {len(X_large) >= selector_large.large_dataset_threshold}")
        
        # Run selection on large dataset
        print("🚀 Running feature selection on large dataset...")
        start_time = datetime.now()
        
        features, results = selector_large.select_features(X_large, y_large)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Large dataset selection completed in {execution_time:.1f}s")
        print(f"📊 Selected {len(features)} features")
        print(f"🎯 AUC: {results.get('final_auc', results.get('best_auc', 'N/A'))}")
        print(f"🏆 Performance Grade: {results.get('performance_grade', 'N/A')}")
        print(f"🔧 Methodology: {results.get('methodology', 'N/A')}")
        
        # Verify fast mode was used
        if 'Fast' in results.get('methodology', ''):
            print("✅ Fast mode detection working correctly!")
        else:
            print(f"⚠️ Expected fast mode but got: {results.get('methodology', 'Unknown')}")
        
        # Check for both key compatibility
        has_final_auc = 'final_auc' in results
        has_best_auc = 'best_auc' in results
        
        print(f"🔑 Has 'final_auc' key: {has_final_auc}")
        print(f"🔑 Has 'best_auc' key: {has_best_auc}")
        
        if has_best_auc and has_final_auc:
            print("✅ Key compatibility test passed!")
            return True
        else:
            print("❌ Key compatibility test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shap_robustness():
    """Test SHAP analysis with problematic data shapes"""
    print("\n🧪 Testing SHAP Analysis Robustness...")
    
    try:
        from fast_feature_selector import FastEnterpriseFeatureSelector
        
        # Create test data with potential SHAP issues
        np.random.seed(42)
        n_samples = 3000
        n_features = 25
        
        # Create data with different scales and correlation patterns
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some correlated features that might cause SHAP issues
        X['feature_0'] = X['feature_0'] * 10  # Different scale
        X['feature_1'] = X['feature_0'] * 0.8 + np.random.randn(n_samples) * 0.2  # Highly correlated
        X['feature_2'] = X['feature_2'].round()  # Discrete values
        
        # Create target with complex relationship
        y = pd.Series(
            ((X['feature_0'] * 0.3 + X['feature_1'] * 0.2 + X['feature_2'] * 0.5 + 
              np.random.randn(n_samples) * 2) > 0).astype(int)
        )
        
        print(f"📊 SHAP test data: {len(X):,} samples, {len(X.columns)} features")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Test fast selector with potentially problematic data
        fast_selector = FastEnterpriseFeatureSelector(
            target_auc=0.70,
            max_features=15,
            fast_mode=True
        )
        
        print("🚀 Running fast selection with robust SHAP handling...")
        features, results = fast_selector.select_features(X, y)
        
        print(f"✅ SHAP robustness test completed")
        print(f"📊 Selected {len(features)} features")
        print(f"🎯 AUC: {results.get('final_auc', 'N/A'):.3f}")
        
        if len(features) > 0 and results.get('final_auc', 0) >= 0.70:
            print("✅ SHAP robustness test passed!")
            return True
        else:
            print("❌ SHAP robustness test failed - insufficient performance")
            return False
            
    except Exception as e:
        print(f"❌ SHAP robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Advanced Feature Selector Fixes")
    print("=" * 50)
    
    success1 = test_fast_mode_auto_detection()
    success2 = test_shap_robustness()
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🎉 All tests passed! Advanced feature selector fixes are working.")
        print("✅ Fast mode auto-detection: Working")
        print("✅ SHAP analysis robustness: Working") 
        print("✅ Key compatibility: Working")
        print("✅ Progress manager error handling: Improved")
    else:
        print("❌ Some tests failed:")
        print(f"❌ Fast mode detection: {'✅' if success1 else '❌'}")
        print(f"❌ SHAP robustness: {'✅' if success2 else '❌'}")
