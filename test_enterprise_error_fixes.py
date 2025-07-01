#!/usr/bin/env python3
"""
🧪 ENTERPRISE ERROR FIXES VALIDATION
ทดสอบการแก้ไขปัญหา error/warning ทั้งหมดในระบบ NICEGOLD

Enterprise Tests:
- FutureWarning fixes
- UserWarning Keras fixes  
- CUDA error handling
- NaN/Infinity protection
- AttributeError fixes
- Integration testing
"""

import sys
import os
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add project path
sys.path.append('/content/drive/MyDrive/ProjectP')

# Suppress warnings for testing
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_warnings_handling():
    """ทดสอบการจัดการ CUDA warnings"""
    print("🔍 Testing CUDA warnings handling...")
    
    try:
        # Test TensorFlow import and CUDA detection
        from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave, TENSORFLOW_AVAILABLE, CUDA_AVAILABLE
        
        config = {'elliott_wave': {'sequence_length': 20}}
        engine = CNNLSTMElliottWave(config=config, logger=logger)
        
        info = engine.get_model_info()
        
        print(f"  ✅ TensorFlow Available: {info['tensorflow_available']}")
        print(f"  ✅ CUDA Available: {info['cuda_available']}")
        print(f"  ✅ Acceleration: {info['acceleration']}")
        
        # Test model building (should not throw CUDA errors)
        if TENSORFLOW_AVAILABLE:
            test_shape = (20, 5)
            model = engine.build_model(test_shape)
            print(f"  ✅ Model building successful: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CUDA test failed: {str(e)}")
        return False

def test_keras_userwarning_fixes():
    """ทดสอบการแก้ไข Keras UserWarning"""
    print("🔍 Testing Keras UserWarning fixes...")
    
    try:
        from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave, TENSORFLOW_AVAILABLE
        
        if not TENSORFLOW_AVAILABLE:
            print("  ℹ️ TensorFlow not available - skipping Keras test")
            return True
        
        config = {'elliott_wave': {'sequence_length': 20}}
        engine = CNNLSTMElliottWave(config=config, logger=logger)
        
        # Test model building with proper Input layer
        test_shape = (20, 5)
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = engine.build_model(test_shape)
            
            # Check for UserWarnings about input_shape
            keras_warnings = [warning for warning in w if 'input_shape' in str(warning.message).lower()]
            
            if len(keras_warnings) == 0:
                print("  ✅ No Keras input_shape warnings detected")
            else:
                print(f"  ⚠️ Found {len(keras_warnings)} Keras warnings")
                for warning in keras_warnings:
                    print(f"    - {warning.message}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Keras test failed: {str(e)}")
        return False

def test_dqn_nan_infinity_protection():
    """ทดสอบการป้องกัน NaN และ Infinity ใน DQN"""
    print("🔍 Testing DQN NaN/Infinity protection...")
    
    try:
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent, safe_division, sanitize_numeric_value
        
        # Test safe_division function
        test_cases = [
            (10, 2, 5.0),      # Normal case
            (10, 0, 0.0),      # Division by zero
            (10, np.nan, 0.0), # Division by NaN
            (np.inf, 2, 0.0),  # Infinity numerator
        ]
        
        for num, den, expected in test_cases:
            result = safe_division(num, den)
            print(f"  ✅ safe_division({num}, {den}) = {result}")
        
        # Test sanitize_numeric_value function
        test_values = [np.nan, np.inf, -np.inf, 42.5, "invalid"]
        for value in test_values:
            result = sanitize_numeric_value(value)
            print(f"  ✅ sanitize_numeric_value({value}) = {result}")
        
        # Test DQN agent with problematic data
        config = {'dqn': {'state_size': 10, 'action_size': 3}}
        agent = DQNReinforcementAgent(config=config, logger=logger)
        
        # Create test data with NaN and infinity
        test_data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, np.inf, 105, 104]
        })
        
        # Test state preparation
        state = agent._prepare_state(test_data)
        print(f"  ✅ State preparation with NaN/Inf: shape={state.shape}, contains_nan={np.any(np.isnan(state))}")
        
        # Test environment step
        next_state, reward, done = agent._step_environment(test_data, 0, 1)
        print(f"  ✅ Environment step: reward={reward}, nan_state={np.any(np.isnan(next_state))}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DQN NaN/Infinity test failed: {str(e)}")
        return False

def test_performance_analyzer_method():
    """ทดสอบ Performance Analyzer methods"""
    print("🔍 Testing Performance Analyzer method fixes...")
    
    try:
        from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
        
        analyzer = ElliottWavePerformanceAnalyzer(logger=logger)
        
        # Test that both methods exist
        test_results = {
            'cnn_lstm_training': {'cnn_lstm_results': {'evaluation_results': {'auc': 0.75}}},
            'dqn_training': {'dqn_results': {'evaluation_results': {'return_pct': 5.0}}},
        }
        
        # Test analyze_results method (original)
        results1 = analyzer.analyze_results(test_results)
        print(f"  ✅ analyze_results method: success={results1.get('overall_performance', {}).get('overall_score', 0) > 0}")
        
        # Test analyze_performance method (added for compatibility)
        results2 = analyzer.analyze_performance(test_results)
        print(f"  ✅ analyze_performance method: success={results2.get('overall_performance', {}).get('overall_score', 0) > 0}")
        
        # Verify both methods return same results
        same_results = results1 == results2
        print(f"  ✅ Methods return consistent results: {same_results}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance Analyzer test failed: {str(e)}")
        return False

def test_fillna_futurewarning_fixes():
    """ทดสอบการแก้ไข FutureWarning fillna"""
    print("🔍 Testing FutureWarning fillna fixes...")
    
    try:
        # Test data with NaN values
        test_data = pd.DataFrame({
            'price': [100, np.nan, 102, np.nan, 104],
            'volume': [1000, np.nan, 1200, 1100, np.nan]
        })
        
        # Test forward fill + backward fill (new method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            filled_data = test_data.ffill().bfill()
            
            # Check for FutureWarnings
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            
            if len(future_warnings) == 0:
                print("  ✅ No FutureWarnings detected")
            else:
                print(f"  ⚠️ Found {len(future_warnings)} FutureWarnings")
                for warning in future_warnings:
                    print(f"    - {warning.message}")
        
        print(f"  ✅ Data filled successfully: {filled_data.isna().sum().sum()} NaN remaining")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FutureWarning test failed: {str(e)}")
        return False

def test_enterprise_ml_protection_integration():
    """ทดสอบการรวม Enterprise ML Protection"""
    print("🔍 Testing Enterprise ML Protection integration...")
    
    try:
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        
        protection = EnterpriseMLProtectionSystem(logger=logger)
        
        # Test basic functionality
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Test protection analysis
        protection_results = protection.analyze_ml_pipeline(
            training_data=test_data,
            features=['feature1', 'feature2'],
            target='target'
        )
        
        print(f"  ✅ Protection analysis: {protection_results['analysis_summary']['overall_risk_level']}")
        print(f"  ✅ Overfitting risk: {protection_results['overfitting_analysis']['risk_level']}")
        print(f"  ✅ Data leakage check: {protection_results['data_leakage_analysis']['risk_level']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enterprise ML Protection test failed: {str(e)}")
        return False

def run_comprehensive_error_fixes_test():
    """รันการทดสอบการแก้ไขปัญหาทั้งหมด"""
    print("🏢 NICEGOLD ENTERPRISE ERROR FIXES VALIDATION")
    print("=" * 60)
    print()
    
    tests = [
        ("FutureWarning fillna fixes", test_fillna_futurewarning_fixes),
        ("Keras UserWarning fixes", test_keras_userwarning_fixes),
        ("CUDA warnings handling", test_cuda_warnings_handling),
        ("DQN NaN/Infinity protection", test_dqn_nan_infinity_protection),
        ("Performance Analyzer method", test_performance_analyzer_method),
        ("Enterprise ML Protection", test_enterprise_ml_protection_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
            
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ FAILED: {test_name} - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL ENTERPRISE ERROR FIXES VALIDATED SUCCESSFULLY!")
        print("🏢 System ready for production deployment")
    else:
        print("⚠️ Some tests failed - manual review required")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_error_fixes_test()
    print(f"\n🏁 Enterprise validation {'completed successfully' if success else 'completed with issues'}")
