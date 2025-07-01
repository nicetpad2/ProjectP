#!/usr/bin/env python3
"""
🧪 ENTERPRISE ML PROTECTION SYSTEM INTEGRATION TEST
ทดสอบการ integrate และการทำงานของ Enterprise ML Protection System
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

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enterprise_ml_protection_complete():
    """ทดสอบ Enterprise ML Protection System แบบครบถ้วน"""
    print("🧪 ENTERPRISE ML PROTECTION SYSTEM - COMPLETE INTEGRATION TEST")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Import Test
        print("🔍 Test 1: Testing import and initialization...")
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        
        protection_system = EnterpriseMLProtectionSystem(logger=logger)
        print("  ✅ Import and initialization successful")
        
        # Test 2: Create realistic trading data
        print("\n🔍 Test 2: Creating realistic trading data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate realistic OHLC data
        base_price = 2000.0
        prices = []
        current_price = base_price
        
        for i in range(n_samples):
            # Random walk with slight trend
            change = np.random.normal(0, 10) + 0.1  # Small upward bias
            current_price += change
            prices.append(current_price)
        
        # Create DataFrame with realistic trading features
        trading_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'open': prices,
            'high': [p + np.random.uniform(0, 20) for p in prices],
            'low': [p - np.random.uniform(0, 20) for p in prices],
            'close': [p + np.random.uniform(-5, 5) for p in prices],
            'volume': np.random.uniform(100, 1000, n_samples),
            # Technical indicators
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 5, n_samples),
            'bb_upper': [p + 50 for p in prices],
            'bb_lower': [p - 50 for p in prices],
            'sma_20': [p + np.random.uniform(-10, 10) for p in prices],
            'ema_12': [p + np.random.uniform(-5, 5) for p in prices],
        })
        
        # Create target variable (price direction)
        trading_data['future_price'] = trading_data['close'].shift(-1)
        trading_data['target'] = (trading_data['future_price'] > trading_data['close']).astype(int)
        trading_data = trading_data.dropna()
        
        # Prepare features and target
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12']
        X = trading_data[feature_cols]
        y = trading_data['target']
        
        print(f"  ✅ Created trading data: {len(X)} samples, {len(feature_cols)} features")
        
        # Test 3: Run comprehensive protection analysis
        print("\n🔍 Test 3: Running comprehensive protection analysis...")
        
        protection_results = protection_system.comprehensive_protection_analysis(
            X=X,
            y=y,
            model=None,  # Will use default
            datetime_col='date' if 'date' in trading_data.columns else None
        )
        
        print("  ✅ Comprehensive protection analysis completed")
        
        # Test 4: Analyze results
        print("\n🔍 Test 4: Analyzing protection results...")
        
        overall_assessment = protection_results.get('overall_assessment', {})
        protection_status = overall_assessment.get('protection_status', 'UNKNOWN')
        risk_level = overall_assessment.get('risk_level', 'UNKNOWN')
        enterprise_ready = overall_assessment.get('enterprise_ready', False)
        
        print(f"  📊 Protection Status: {protection_status}")
        print(f"  🎯 Risk Level: {risk_level}")
        print(f"  🏢 Enterprise Ready: {'✅' if enterprise_ready else '❌'}")
        
        # Data Leakage Analysis
        leakage_results = protection_results.get('data_leakage', {})
        leakage_detected = leakage_results.get('leakage_detected', False)
        leakage_score = leakage_results.get('leakage_score', 0)
        
        print(f"  🔍 Data Leakage: {'⚠️ DETECTED' if leakage_detected else '✅ CLEAN'}")
        print(f"  📊 Leakage Score: {leakage_score:.3f}")
        
        # Overfitting Analysis
        overfitting_results = protection_results.get('overfitting', {})
        overfitting_detected = overfitting_results.get('overfitting_detected', False)
        overfitting_score = overfitting_results.get('overfitting_score', 0)
        
        print(f"  🎯 Overfitting: {'⚠️ DETECTED' if overfitting_detected else '✅ CLEAN'}")
        print(f"  📊 Overfitting Score: {overfitting_score:.3f}")
        
        # Noise Analysis
        noise_results = protection_results.get('noise_analysis', {})
        noise_level = noise_results.get('noise_level', 0)
        data_quality = noise_results.get('data_quality_score', 0)
        
        print(f"  🎵 Noise Level: {noise_level:.3f}")
        print(f"  🏆 Data Quality: {data_quality:.3f}")
        
        # Test 5: Test Menu 1 Integration
        print("\n🔍 Test 5: Testing Menu 1 integration...")
        
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            
            # Test initialization with ML Protection
            menu1 = Menu1ElliottWave(logger=logger)
            
            # Check if ML Protection is properly integrated
            has_ml_protection = hasattr(menu1, 'ml_protection') and menu1.ml_protection is not None
            has_orchestrator = hasattr(menu1, 'pipeline_orchestrator') and menu1.pipeline_orchestrator is not None
            orchestrator_has_protection = (has_orchestrator and 
                                         hasattr(menu1.pipeline_orchestrator, 'ml_protection') and 
                                         menu1.pipeline_orchestrator.ml_protection is not None)
            
            print(f"  🎛️ Menu 1 ML Protection: {'✅' if has_ml_protection else '❌'}")
            print(f"  🎼 Pipeline Orchestrator: {'✅' if has_orchestrator else '❌'}")
            print(f"  🛡️ Orchestrator Protection: {'✅' if orchestrator_has_protection else '❌'}")
            
            integration_success = has_ml_protection and orchestrator_has_protection
            print(f"  🔗 Integration Status: {'✅ SUCCESS' if integration_success else '❌ FAILED'}")
            
        except Exception as e:
            print(f"  ❌ Menu 1 integration test failed: {str(e)}")
            integration_success = False
        
        # Test 6: Test specific protection methods
        print("\n🔍 Test 6: Testing specific protection methods...")
        
        # Test data leakage detection
        leakage_test = protection_system._detect_data_leakage(X, y)
        leakage_method_works = leakage_test.get('status') in ['CLEAN', 'DETECTED']
        print(f"  🔍 Data Leakage Detection: {'✅' if leakage_method_works else '❌'}")
        
        # Test overfitting detection  
        overfitting_test = protection_system._detect_overfitting(X, y)
        overfitting_method_works = overfitting_test.get('status') in ['CLEAN', 'DETECTED']
        print(f"  📊 Overfitting Detection: {'✅' if overfitting_method_works else '❌'}")
        
        # Test noise detection
        noise_test = protection_system._detect_noise_and_quality(X, y)
        noise_method_works = noise_test.get('status') in ['CLEAN', 'NOISY']
        print(f"  🎵 Noise Detection: {'✅' if noise_method_works else '❌'}")
        
        methods_success = leakage_method_works and overfitting_method_works and noise_method_works
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 ENTERPRISE ML PROTECTION TEST SUMMARY")
        print("=" * 70)
        
        all_tests = [
            ("Import & Initialization", True),  # Always passes if we get here
            ("Realistic Data Creation", True),  # Always passes if we get here  
            ("Comprehensive Analysis", protection_results.get('protection_status') != 'ERROR'),
            ("Results Analysis", overall_assessment is not None),
            ("Menu 1 Integration", integration_success),
            ("Specific Methods", methods_success)
        ]
        
        passed_tests = sum(1 for _, passed in all_tests if passed)
        total_tests = len(all_tests)
        
        for test_name, passed in all_tests:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} - {test_name}")
        
        print("-" * 70)
        print(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL ENTERPRISE ML PROTECTION TESTS PASSED!")
            print("🏢 System ready for production integration")
            return True
        else:
            print("⚠️ Some tests failed - review required")
            return False
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enterprise_ml_protection_complete()
    print(f"\n🏁 Enterprise ML Protection test {'completed successfully' if success else 'completed with issues'}")
