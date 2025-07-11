#!/usr/bin/env python3
"""
ทดสอบการ initialize EnhancedMenu1ElliottWave
เพื่อตรวจสอบปัญหาใน constructor และ component initialization
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_menu1_initialization():
    """ทดสอบการ initialize Menu 1"""
    print("🧪 Testing EnhancedMenu1ElliottWave Initialization")
    print("="*60)
    
    try:
        print("📝 Step 1: Importing required modules...")
        from core.unified_enterprise_logger import get_unified_logger
        from core.config import get_global_config
        print("✅ Core modules imported successfully")
        
        print("\n📝 Step 2: Importing EnhancedMenu1ElliottWave...")
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        print("✅ EnhancedMenu1ElliottWave imported successfully")
        
        print("\n📝 Step 3: Getting configuration...")
        config = get_global_config().config
        print(f"✅ Configuration loaded: {type(config)}")
        
        print("\n📝 Step 4: Creating EnhancedMenu1ElliottWave instance...")
        print("  🔄 Calling constructor...")
        menu1 = EnhancedMenu1ElliottWave(config=config)
        print("✅ Instance created successfully!")
        
        print("\n📝 Step 5: Checking component status...")
        components = {
            'data_processor': menu1.data_processor,
            'model_manager': menu1.model_manager,
            'feature_selector': menu1.feature_selector,
            'cnn_lstm_engine': menu1.cnn_lstm_engine,
            'dqn_agent': menu1.dqn_agent,
            'performance_analyzer': menu1.performance_analyzer,
            'ml_protection': menu1.ml_protection
        }
        
        for name, component in components.items():
            status = "✅ Initialized" if component is not None else "❌ None"
            print(f"  {name}: {status}")
        
        # Count initialized components
        initialized_count = sum(1 for c in components.values() if c is not None)
        total_count = len(components)
        
        print(f"\n📊 Summary: {initialized_count}/{total_count} components initialized")
        
        if initialized_count == 0:
            print("❌ NO COMPONENTS INITIALIZED - This is the problem!")
            return False
        elif initialized_count < total_count:
            print("⚠️ PARTIAL INITIALIZATION - Some components missing")
            return False
        else:
            print("✅ ALL COMPONENTS INITIALIZED - Ready for pipeline")
            return True
            
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_menu1_initialization()
    
    print("\n" + "="*60)
    if success:
        print("🎉 TEST PASSED: Initialization working correctly")
    else:
        print("🚨 TEST FAILED: Initialization has issues")
    print("="*60) 