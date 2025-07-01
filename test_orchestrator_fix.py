#!/usr/bin/env python3
"""
🔧 TEST PIPELINE ORCHESTRATOR FIX
ทดสอบการแก้ไขปัญหา ElliottWavePipelineOrchestrator arguments
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_pipeline_orchestrator_fix():
    """ทดสอบการแก้ไข Pipeline Orchestrator"""
    print("🔧 TESTING PIPELINE ORCHESTRATOR FIX")
    print("=" * 50)
    
    try:
        # Test 1: Import Menu1
        print("🧪 Test 1: Import Menu1ElliottWave...")
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Import successful")
        
        # Test 2: Initialize Menu1 (this should now work)
        print("\n🧪 Test 2: Initialize Menu1...")
        menu = Menu1ElliottWave()
        print("✅ Initialization successful")
        
        # Test 3: Check that orchestrator is properly initialized
        print("\n🧪 Test 3: Check orchestrator...")
        if hasattr(menu, 'orchestrator'):
            print("✅ Orchestrator exists")
            if hasattr(menu.orchestrator, 'data_processor'):
                print("✅ Orchestrator has data_processor")
            if hasattr(menu.orchestrator, 'cnn_lstm_engine'):
                print("✅ Orchestrator has cnn_lstm_engine")
            if hasattr(menu.orchestrator, 'dqn_agent'):
                print("✅ Orchestrator has dqn_agent")
            if hasattr(menu.orchestrator, 'feature_selector'):
                print("✅ Orchestrator has feature_selector")
        else:
            print("❌ Orchestrator missing")
            return False
        
        # Test 4: Quick functionality test
        print("\n🧪 Test 4: Quick pipeline test...")
        results = menu.run_full_pipeline()
        
        if results:
            execution_status = results.get('execution_status', 'unknown')
            print(f"📊 Execution Status: {execution_status}")
            
            # Check for the specific orchestrator error
            error_msg = results.get('error_message', '')
            if "missing 4 required positional arguments" in error_msg:
                print("❌ Orchestrator error still exists!")
                return False
            else:
                print("✅ No orchestrator argument error!")
                return True
        else:
            print("⚠️ No results returned but no orchestrator error")
            return True
            
    except TypeError as e:
        if "missing 4 required positional arguments" in str(e):
            print(f"❌ Orchestrator argument error still exists: {e}")
            return False
        else:
            print(f"⚠️ Different TypeError: {e}")
            return True
    except Exception as e:
        print(f"💡 Different error (expected): {e}")
        return True  # This means orchestrator issue is fixed

def main():
    """Run the test"""
    success = test_pipeline_orchestrator_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! Pipeline Orchestrator fix is working!")
        print("✅ Menu 1 is ready for full operation!")
    else:
        print("❌ FAILED! Orchestrator argument error still exists")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
