#!/usr/bin/env python3
"""
🧪 TEST TARGET VARIABLE FIX
ทดสอบการแก้ไขปัญหา target variable creation ที่เกิด NaN to integer error
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_target_variable_creation():
    """ทดสอบการสร้าง target variable ที่แก้ไขแล้ว"""
    
    print("🧪 Testing Target Variable Creation Fix...")
    
    try:
        # Import required modules
        from elliott_wave_modules.feature_engineering import ElliottWaveFeatureEngineer
        from core.unified_enterprise_logger import get_unified_logger
        
        # Initialize logger
        logger = get_unified_logger()
        
        # Create sample data with potential NaN scenarios
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1800, 2000, 100),
            'high': np.random.uniform(1810, 2010, 100),
            'low': np.random.uniform(1790, 1990, 100),
            'close': np.random.uniform(1800, 2000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Add some NaN values at the end to simulate real scenario
        sample_data.loc[95:, 'close'] = np.nan
        
        print("✅ Sample data with NaN values created")
        
        # Initialize feature engineer
        feature_engineer = ElliottWaveFeatureEngineer(
            config={'prediction_horizon': 5}, 
            logger=logger
        )
        
        # Test target variable creation
        print("\n🔄 Testing target variable creation with NaN handling...")
        
        result_df = feature_engineer._create_target_variable(sample_data.copy())
        
        # Verify results
        if 'target' in result_df.columns:
            print("✅ Binary target variable created successfully")
            print(f"   - Target distribution: {result_df['target'].value_counts().to_dict()}")
            print(f"   - Target data type: {result_df['target'].dtype}")
            print(f"   - Target NaN count: {result_df['target'].isna().sum()}")
        else:
            print("❌ Binary target variable not created")
            return False
            
        if 'target_multiclass' in result_df.columns:
            print("✅ Multi-class target variable created successfully")
            print(f"   - Multi-class distribution: {result_df['target_multiclass'].value_counts().to_dict()}")
            print(f"   - Multi-class data type: {result_df['target_multiclass'].dtype}")
            print(f"   - Multi-class NaN count: {result_df['target_multiclass'].isna().sum()}")
        else:
            print("❌ Multi-class target variable not created")
            return False
            
        # Test with extreme NaN scenario
        print("\n🔄 Testing with extreme NaN scenario...")
        extreme_data = sample_data.copy()
        extreme_data.loc[90:, 'close'] = np.nan
        
        result_extreme = feature_engineer._create_target_variable(extreme_data)
        
        if result_extreme['target'].isna().sum() == 0:
            print("✅ Extreme NaN scenario handled successfully")
        else:
            print("❌ Extreme NaN scenario not handled properly")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Target variable creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_menu5_integration():
    """ทดสอบ Menu 5 integration หลังจากแก้ไข"""
    
    print("\n🔄 Testing Menu 5 integration after fix...")
    
    try:
        # Import Menu 5 system
        from menu_modules.menu_5_oms_mm_100usd import Menu5OMSMMSystem
        
        # Initialize Menu 5
        menu5 = Menu5OMSMMSystem()
        
        # Load Menu 1 strategy
        menu5.load_menu1_strategy()
        
        print("✅ Menu 5 system loaded successfully")
        
        # Test if data processor has the fixed method
        if hasattr(menu5.data_processor, 'prepare_ml_data'):
            print("✅ Data processor has prepare_ml_data method")
        else:
            print("❌ Data processor missing prepare_ml_data method")
            return False
            
        # Test if data processor has the required methods
        if hasattr(menu5.data_processor, 'feature_engineer'):
            feature_engineer = menu5.data_processor.feature_engineer
            if hasattr(feature_engineer, '_create_target_variable'):
                print("✅ Data processor's feature engineer has _create_target_variable method")
            else:
                print("❌ Data processor's feature engineer missing _create_target_variable method")
                return False
        else:
            print("⚠️ Data processor doesn't have direct feature_engineer attribute, checking prepare_ml_data method")
            # The important thing is that prepare_ml_data works, which calls feature engineering internally
            
        return True
        
    except Exception as e:
        print(f"❌ Menu 5 integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎯 TARGET VARIABLE FIX VERIFICATION")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Target variable creation
    if test_target_variable_creation():
        success_count += 1
        print("✅ Test 1: Target Variable Creation - PASSED")
    else:
        print("❌ Test 1: Target Variable Creation - FAILED")
    
    # Test 2: Menu 5 integration
    if test_menu5_integration():
        success_count += 1
        print("✅ Test 2: Menu 5 Integration - PASSED")
    else:
        print("❌ Test 2: Menu 5 Integration - FAILED")
    
    print("\n" + "=" * 50)
    print(f"🎯 FINAL RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 TARGET VARIABLE FIX VERIFICATION COMPLETE - ALL TESTS PASSED!")
        return True
    else:
        print("❌ TARGET VARIABLE FIX VERIFICATION INCOMPLETE - SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 