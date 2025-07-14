#!/usr/bin/env python3
"""
🧪 TEST DATA PROCESSOR FIX
ทดสอบการแก้ไข prepare_ml_data method ใน ElliottWaveDataProcessor
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_data_processor_fix():
    """ทดสอบ ElliottWaveDataProcessor prepare_ml_data method"""
    
    print("🧪 Testing ElliottWaveDataProcessor prepare_ml_data method...")
    
    try:
        # Import required modules
        from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
        from core.unified_enterprise_logger import get_unified_logger
        
        # Initialize logger
        logger = get_unified_logger()
        
        # Initialize data processor
        data_processor = ElliottWaveDataProcessor(logger=logger)
        
        print("✅ ElliottWaveDataProcessor initialized successfully")
        
        # Test 1: Check if prepare_ml_data method exists
        if hasattr(data_processor, 'prepare_ml_data'):
            print("✅ prepare_ml_data method exists")
        else:
            print("❌ prepare_ml_data method not found")
            return False
        
        # Test 2: Create sample data
        print("\n🔄 Creating sample data...")
        
        # Create sample OHLC data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 2000.0
        price_changes = np.random.normal(0, 0.002, 1000)
        close_prices = base_price * np.exp(np.cumsum(price_changes))
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.normal(0, 0.001, 1000)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
            'close': close_prices,
            'volume': np.random.randint(100, 1000, 1000)
        })
        
        print(f"✅ Sample data created: {len(sample_data)} rows")
        
        # Test 3: Process data using process_data_for_elliott_wave
        print("\n🔄 Processing data for Elliott Wave...")
        
        processed_data = data_processor.process_data_for_elliott_wave(sample_data)
        
        if processed_data is not None:
            print(f"✅ Data processed successfully: {processed_data.shape}")
            print(f"✅ Columns: {list(processed_data.columns)}")
        else:
            print("❌ Data processing failed")
            return False
        
        # Test 4: Test prepare_ml_data method
        print("\n🧠 Testing prepare_ml_data method...")
        
        try:
            X, y = data_processor.prepare_ml_data(processed_data)
            
            print(f"✅ prepare_ml_data successful!")
            print(f"✅ X shape: {X.shape}")
            print(f"✅ y shape: {y.shape}")
            print(f"✅ Feature columns: {list(X.columns)}")
            print(f"✅ Target distribution: {y.value_counts().to_dict()}")
            
            # Validate data types
            print(f"✅ X data types: {X.dtypes.value_counts().to_dict()}")
            print(f"✅ y data type: {y.dtype}")
            
            return True
            
        except Exception as e:
            print(f"❌ prepare_ml_data failed: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_menu5_integration():
    """ทดสอบการ integrate กับ Menu 5"""
    
    print("\n🔄 Testing Menu 5 integration...")
    
    try:
        # Test import Menu 5 class
        from menu_modules.menu_5_oms_mm_100usd import Menu5OMSMMSystem
        
        print("✅ Menu5OMSMMSystem imported successfully")
        
        # Initialize Menu 5
        menu5 = Menu5OMSMMSystem()
        
        print("✅ Menu5OMSMMSystem initialized successfully")
        
        # Load Menu 1 strategy to initialize data_processor properly
        menu5.load_menu1_strategy()
        
        print("✅ Menu 1 strategy loaded successfully")
        
        # Test data processor has prepare_ml_data method
        if menu5.data_processor and hasattr(menu5.data_processor, 'prepare_ml_data'):
            print("✅ Menu 5 data processor has prepare_ml_data method")
        else:
            print("❌ Menu 5 data processor missing prepare_ml_data method")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Menu 5 integration test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🧪 TESTING DATA PROCESSOR FIX")
    print("=" * 50)
    
    # Test 1: Data processor fix
    test1_result = test_data_processor_fix()
    
    # Test 2: Menu 5 integration
    test2_result = test_menu5_integration()
    
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS:")
    print(f"✅ Data Processor Fix: {'PASSED' if test1_result else 'FAILED'}")
    print(f"✅ Menu 5 Integration: {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ prepare_ml_data method is working correctly")
        print("✅ Menu 5 integration is ready")
    else:
        print("\n❌ Some tests failed!")
        print("❌ Please check the errors above") 