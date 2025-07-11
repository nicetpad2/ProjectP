#!/usr/bin/env python3
"""
ทดสอบ Menu 1 Elliott Wave Pipeline
เพื่อตรวจสอบว่าสามารถทำงานครบทุก step ได้หรือไม่
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_menu1_pipeline():
    """ทดสอบ Menu 1 Pipeline"""
    print("🧪 Testing Elliott Wave Full Pipeline")
    print("="*60)
    
    try:
        print("📝 Step 1: Importing and setting up...")
        from core.unified_enterprise_logger import get_unified_logger
        from core.config import get_global_config
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        print("✅ All modules imported successfully")
        
        print("\n📝 Step 2: Creating Menu 1 instance...")
        config = get_global_config().config
        menu1 = EnhancedMenu1ElliottWave(config=config)
        print("✅ Menu 1 instance created successfully")
        
        print("\n📝 Step 3: Verifying components are initialized...")
        components = {
            'data_processor': menu1.data_processor,
            'model_manager': menu1.model_manager,
            'feature_selector': menu1.feature_selector,
        }
        
        all_good = True
        for name, component in components.items():
            if component is None:
                print(f"❌ {name}: Not initialized")
                all_good = False
            else:
                print(f"✅ {name}: Ready")
        
        if not all_good:
            print("❌ Some components not ready - cannot run pipeline")
            return False
        
        print("\n📝 Step 4: Running Elliott Wave Pipeline...")
        print("🚀 Starting pipeline execution...")
        
        # Run the pipeline
        result = menu1.run()
        
        print("\n📝 Step 5: Analyzing results...")
        if result:
            if result.get('status') == 'ERROR':
                print(f"❌ Pipeline failed: {result.get('message', 'Unknown error')}")
                return False
            else:
                print("✅ Pipeline completed!")
                
                # Check if we have meaningful results
                if 'session_summary' in result:
                    summary = result['session_summary']
                    print(f"📊 Steps completed: {summary.get('total_steps', 'N/A')}")
                    print(f"🎯 Features selected: {summary.get('selected_features', 'N/A')}")
                    print(f"🧠 Model AUC: {summary.get('model_auc', 'N/A')}")
                
                return True
        else:
            print("⚠️ Pipeline returned no results")
            return False
            
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_menu1_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE TEST PASSED: Menu 1 works completely!")
        print("✅ All steps executed successfully")
    else:
        print("🚨 PIPELINE TEST FAILED: Menu 1 has issues")
        print("❌ Pipeline did not complete successfully")
    print("="*60) 