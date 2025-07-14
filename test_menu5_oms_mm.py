#!/usr/bin/env python3
"""
Test script for Menu 5 OMS & MM System
ทดสอบเมนูที่ 5 ระบบ OMS & MM ด้วยทุน 100 USD
"""

import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_menu5_oms_mm():
    """ทดสอบเมนูที่ 5 OMS & MM System"""
    print("🏢 Testing Menu 5: OMS & MM System")
    print("=" * 60)
    
    try:
        print("🔄 Step 1: Testing import...")
        from menu_modules.menu_5_oms_mm_100usd import Menu5OMSMMSystem, run_menu_5_oms_mm
        print("✅ Import successful")
        
        print("\n🔄 Step 2: Testing system initialization...")
        system = Menu5OMSMMSystem()
        print("✅ System initialized")
        
        print("\n🔄 Step 3: Testing Menu 1 strategy loading...")
        if system.load_menu1_strategy():
            print("✅ Menu 1 strategy loaded")
        else:
            print("⚠️ Menu 1 strategy loading failed")
        
        print("\n🔄 Step 4: Testing full system run...")
        print("⚠️ This will take several minutes...")
        
        # Run the complete system
        results = run_menu_5_oms_mm()
        
        if results:
            print("\n🎉 Menu 5 OMS & MM System test completed successfully!")
            print(f"💰 Final Capital: ${results.get('final_capital', 100):.2f}")
            print(f"📈 Total Return: {results.get('total_return_pct', 0):.2f}%")
            print(f"🎯 Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"🏪 Total Trades: {results.get('trades_executed', 0)}")
        else:
            print("❌ Menu 5 system test failed")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_menu5_oms_mm() 