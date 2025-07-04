#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK FIX TEST FOR PERFORMANCE OPTIMIZATION
Simple test to verify the optimization system works without errors
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Environment optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

# Suppress warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create simple test data"""
    print("📊 Creating test data...")
    
    # Create synthetic data that resembles real trading data
    np.random.seed(42)
    n_samples = 5000  # Smaller dataset for testing
    n_features = 20   # Fewer features for testing
    
    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create binary target
    y = pd.Series(np.random.binomial(1, 0.3, n_samples), name='target')
    
    print(f"✅ Test data created: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def test_original_feature_selector():
    """Test original feature selector"""
    print("\n🧪 Testing original feature selector...")
    
    try:
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        
        X, y = create_test_data()
        
        # Create feature selector with very conservative settings
        selector = EnterpriseShapOptunaFeatureSelector(
            target_auc=0.60,  # Lower target for testing
            max_features=10,
            n_trials=5,      # Very few trials
            timeout=30       # Short timeout
        )
        
        print("⚡ Running feature selection...")
        selected_features, results = selector.select_features(X, y)
        
        print(f"✅ Feature selection completed!")
        print(f"🎯 Selected features: {len(selected_features)}")
        if 'best_auc' in results:
            print(f"📈 AUC achieved: {results['best_auc']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Original feature selector failed: {e}")
        return False

def test_optimization_engine():
    """Test the new optimization engine"""
    print("\n🚀 Testing optimization engine...")
    
    try:
        from nicegold_resource_optimization_engine import NiceGoldResourceOptimizationEngine
        
        X, y = create_test_data()
        
        # Create optimization engine
        engine = NiceGoldResourceOptimizationEngine()
        
        print("⚡ Running optimized pipeline...")
        results = engine.execute_optimized_pipeline(X, y)
        
        print(f"✅ Optimization completed!")
        
        if 'selected_features' in results:
            print(f"🎯 Selected features: {len(results['selected_features'])}")
        
        if 'execution_metrics' in results:
            exec_time = results['execution_metrics'].get('execution_time_seconds', 0)
            print(f"⏱️ Execution time: {exec_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 QUICK PERFORMANCE OPTIMIZATION TEST")
    print("="*50)
    
    # Test original system
    original_works = test_original_feature_selector()
    
    # Test optimization engine
    optimization_works = test_optimization_engine()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Original Feature Selector: {'✅ PASS' if original_works else '❌ FAIL'}")
    print(f"Optimization Engine:       {'✅ PASS' if optimization_works else '❌ FAIL'}")
    
    overall_success = original_works and optimization_works
    print(f"\n🏆 OVERALL: {'✅ SUCCESS' if overall_success else '❌ NEEDS FIX'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
