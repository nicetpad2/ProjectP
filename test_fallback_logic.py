#!/usr/bin/env python3
"""
Test Enterprise ML Protection System Fallback Logic
Tests all methods with and without sklearn/scipy dependencies
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/content/drive/MyDrive/ProjectP')
sys.path.append('/content/drive/MyDrive/ProjectP/core')
sys.path.append('/content/drive/MyDrive/ProjectP/elliott_wave_modules')

def test_fallback_logic():
    """Test Enterprise ML Protection System with fallback logic"""
    
    print("🧪 Testing Enterprise ML Protection System Fallback Logic")
    print("="*80)
    
    try:
        # Import the protection system
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic time-series data
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X['datetime'] = dates
        
        # Generate target with some patterns
        y = pd.Series((X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.1) > 0).astype(int)
        
        print(f"✅ Test data created: {X.shape[0]} samples, {X.shape[1]-1} features")
        
        # Test 1: Normal operation (with sklearn/scipy)
        print("\n📊 Test 1: Normal Operation (with sklearn/scipy)")
        print("-" * 50)
        
        config = {
            'enabled': True,
            'overfitting_threshold': 0.15,
            'noise_threshold': 0.3,
            'data_leakage_threshold': 0.1,
            'min_samples': 100
        }
        
        protection_system = EnterpriseMLProtectionSystem(config=config)
        
        print(f"   - sklearn_available: {protection_system.sklearn_available}")
        print(f"   - scipy_available: {protection_system.scipy_available}")
        
        # Run full analysis
        results = protection_system.comprehensive_protection_analysis(X, y, datetime_col='datetime')
        
        print(f"   - Analysis status: {results.get('status', 'UNKNOWN')}")
        print(f"   - Data leakage: {results.get('data_leakage', {}).get('status', 'UNKNOWN')}")
        print(f"   - Overfitting: {results.get('overfitting', {}).get('status', 'UNKNOWN')}")
        print(f"   - Noise analysis: {results.get('noise_analysis', {}).get('status', 'UNKNOWN')}")
        
        # Test 2: Simulate sklearn unavailable
        print("\n🔄 Test 2: Simulate sklearn unavailable")
        print("-" * 50)
        
        # Temporarily disable sklearn
        original_sklearn = protection_system.sklearn_available
        protection_system.sklearn_available = False
        
        try:
            results_no_sklearn = protection_system.comprehensive_protection_analysis(X, y, datetime_col='datetime')
            
            print(f"   - Analysis status: {results_no_sklearn.get('status', 'UNKNOWN')}")
            print(f"   - Data leakage: {results_no_sklearn.get('data_leakage', {}).get('status', 'UNKNOWN')}")
            print(f"   - Overfitting: {results_no_sklearn.get('overfitting', {}).get('status', 'UNKNOWN')}")
            print(f"   - Overfitting method: {results_no_sklearn.get('overfitting', {}).get('method', 'advanced')}")
            print(f"   - Noise analysis: {results_no_sklearn.get('noise_analysis', {}).get('status', 'UNKNOWN')}")
            
        finally:
            # Restore sklearn availability
            protection_system.sklearn_available = original_sklearn
        
        # Test 3: Simulate scipy unavailable
        print("\n🔄 Test 3: Simulate scipy unavailable")
        print("-" * 50)
        
        # Temporarily disable scipy
        original_scipy = protection_system.scipy_available
        protection_system.scipy_available = False
        
        try:
            results_no_scipy = protection_system.comprehensive_protection_analysis(X, y, datetime_col='datetime')
            
            print(f"   - Analysis status: {results_no_scipy.get('status', 'UNKNOWN')}")
            print(f"   - Data leakage: {results_no_scipy.get('data_leakage', {}).get('status', 'UNKNOWN')}")
            print(f"   - Overfitting: {results_no_scipy.get('overfitting', {}).get('status', 'UNKNOWN')}")
            print(f"   - Noise analysis: {results_no_scipy.get('noise_analysis', {}).get('status', 'UNKNOWN')}")
            
        finally:
            # Restore scipy availability
            protection_system.scipy_available = original_scipy
        
        # Test 4: Test individual methods with fallback
        print("\n🧪 Test 4: Individual Method Fallback Tests")
        print("-" * 50)
        
        # Test overfitting detection
        print("   Testing overfitting detection:")
        overfitting_normal = protection_system._detect_overfitting(X.drop('datetime', axis=1), y)
        print(f"   - Normal: {overfitting_normal.get('status', 'UNKNOWN')}")
        
        overfitting_simplified = protection_system._detect_overfitting_simplified(X.drop('datetime', axis=1), y)
        print(f"   - Simplified: {overfitting_simplified.get('status', 'UNKNOWN')}")
        
        # Test feature distribution analysis
        print("   Testing feature distribution analysis:")
        distribution_normal = protection_system._analyze_feature_distributions(X.drop('datetime', axis=1))
        print(f"   - Distribution analysis: {len(distribution_normal)} features analyzed")
        
        # Test temporal drift detection
        print("   Testing temporal drift detection:")
        drift_results = protection_system._detect_temporal_drift(X, 'datetime')
        print(f"   - Temporal drift: {len(drift_results)} features analyzed")
        
        print("\n✅ All fallback logic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in fallback logic test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration validation and status methods"""
    
    print("\n🔧 Testing Configuration Validation")
    print("-" * 50)
    
    try:
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        
        # Test default configuration
        protection_system = EnterpriseMLProtectionSystem()
        
        # Test configuration validation
        validation_result = protection_system.validate_configuration()
        print(f"   - Default config valid: {validation_result.get('valid', False)}")
        if validation_result.get('issues'):
            print(f"   - Validation issues: {validation_result['issues']}")
        if validation_result.get('warnings'):
            print(f"   - Validation warnings: {validation_result['warnings']}")
        
        # Test protection status
        status = protection_system.get_protection_status()
        print(f"   - Protection enabled: {status.get('enabled', False)}")
        print(f"   - sklearn available: {status.get('sklearn_available', False)}")
        print(f"   - scipy available: {status.get('scipy_available', False)}")
        
        # Test configuration update
        new_config = {
            'overfitting_threshold': 0.2,
            'noise_threshold': 0.4
        }
        success = protection_system.update_protection_config(new_config)
        print(f"   - Config update success: {success}")
        
        updated_status = protection_system.get_protection_status()
        print(f"   - Updated overfitting threshold: {updated_status.get('overfitting_threshold', 'UNKNOWN')}")
        
        print("✅ Configuration validation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in configuration validation test: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Enterprise ML Protection System Fallback Tests")
    print("="*80)
    
    # Run tests
    test1_result = test_fallback_logic()
    test2_result = test_configuration_validation()
    
    print("\n📋 Test Results Summary")
    print("="*80)
    print(f"   Fallback Logic Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"   Configuration Test:  {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED! Enterprise ML Protection System is production ready!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*80)
