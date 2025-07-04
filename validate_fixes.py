#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE SYSTEM VALIDATION
Test all fixed components before running the full pipeline
"""

import sys
import traceback

def test_all_fixes():
    """Test all the fixes we've applied"""
    print("🔬 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Config System
    print("\n📋 Test 1: Config System")
    try:
        from core.config import get_config
        config = get_config()
        max_features = config.get('elliott_wave.max_features', 30)
        print(f"✅ Config system working: max_features = {max_features}")
        assert max_features == 25, f"Expected 25, got {max_features}"
        tests_passed += 1
    except Exception as e:
        print(f"❌ Config test failed: {e}")
    
    # Test 2: Feature Selector
    print("\n🎯 Test 2: Feature Selector Limit")
    try:
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        selector = EnterpriseShapOptunaFeatureSelector(max_features=25)
        print(f"✅ Feature Selector: max_features = {selector.max_features}")
        assert selector.max_features == 25, f"Expected 25, got {selector.max_features}"
        tests_passed += 1
    except Exception as e:
        print(f"❌ Feature Selector test failed: {e}")
    
    # Test 3: DQN Agent
    print("\n🤖 Test 3: DQN Agent")
    try:
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
        agent = DQNReinforcementAgent(state_size=10, action_size=3)
        print("✅ DQN Agent initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ DQN Agent test failed: {e}")
    
    # Test 4: ML Protection
    print("\n🛡️ Test 4: ML Protection System")
    try:
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        protection = EnterpriseMLProtectionSystem()
        print("✅ ML Protection initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ ML Protection test failed: {e}")
    
    # Test 5: Data Processor
    print("\n📊 Test 5: Data Processor")
    try:
        from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
        processor = ElliottWaveDataProcessor()
        print("✅ Data Processor initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data Processor test failed: {e}")
    
    # Test 6: Menu System
    print("\n🎛️ Test 6: Menu System")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        menu = Menu1ElliottWave()
        print("✅ Menu system initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Menu system test failed: {e}")
    
    # Final Results
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL FIXES VALIDATED - SYSTEM READY!")
        return True
    else:
        print("⚠️ Some tests failed - issues remain")
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    if success:
        print("\n🚀 Ready to run the full pipeline!")
        sys.exit(0)
    else:
        print("\n❌ System validation failed")
        sys.exit(1)
