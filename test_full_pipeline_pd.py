#!/usr/bin/env python3
"""
🧪 TEST: ทดสอบการรัน Full Pipeline จริงเพื่อหาปัญหา pd is not defined
"""

import sys
import os
from pathlib import Path

# Add project path
sys.path.append('/content/drive/MyDrive/ProjectP')

print("🧪 TESTING: Full Pipeline execution")
print("="*60)

try:
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
    print("✅ Menu1ElliottWaveFixed imported")
    
    menu1 = Menu1ElliottWaveFixed()
    print("✅ Menu1ElliottWaveFixed initialized")
    
    print("\n🚀 Running full pipeline...")
    print("This will test the actual execution where pd error might occur")
    
    # รัน pipeline จริง
    result = menu1.run_full_pipeline()
    
    if result and result.get('status') == 'success':
        print("✅ Pipeline executed successfully!")
        print(f"📊 Overall Score: {result.get('overall_score', 'N/A')}")
        print(f"🎯 AUC Score: {result.get('auc_score', 'N/A')}")
        print(f"🤖 DQN Reward: {result.get('dqn_reward', 'N/A')}")
        print(f"✅ Enterprise Compliant: {result.get('enterprise_compliant', False)}")
    else:
        print("❌ Pipeline execution failed")
        print(f"Error: {result}")

except Exception as e:
    print(f"❌ Pipeline execution failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 SUMMARY: Pipeline Test Complete")
print("="*60)
