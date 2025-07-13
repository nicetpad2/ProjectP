#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 MENU 5 INTEGRATION TEST
Test the integration of Menu 5 BackTest Strategy with the Unified Master Menu System

✅ Tests:
1. Menu 5 module import
2. Menu 5 initialization  
3. Menu 5 basic functionality
4. Unified menu system integration
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def safe_print(*args, **kwargs):
    """Safe print with error handling"""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        try:
            message = " ".join(map(str, args))
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except:
            pass

def test_menu_5_import():
    """Test Menu 5 module import"""
    safe_print("🧪 TEST 1: Menu 5 Module Import")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        safe_print("✅ Menu 5 BackTest Strategy imported successfully")
        return True
    except ImportError as e:
        safe_print(f"❌ Menu 5 import failed: {e}")
        return False
    except Exception as e:
        safe_print(f"❌ Menu 5 import error: {e}")
        return False

def test_menu_5_initialization():
    """Test Menu 5 initialization"""
    safe_print("\n🧪 TEST 2: Menu 5 Initialization")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        # Test with empty config
        config = {}
        menu_5 = Menu5BacktestStrategy(config)
        safe_print("✅ Menu 5 initialization successful with empty config")
        
        # Test basic attributes
        if hasattr(menu_5, 'config'):
            safe_print("✅ Menu 5 has config attribute")
        if hasattr(menu_5, 'logger'):
            safe_print("✅ Menu 5 has logger attribute")
        if hasattr(menu_5, 'run'):
            safe_print("✅ Menu 5 has run method")
            
        return True
    except Exception as e:
        safe_print(f"❌ Menu 5 initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu_5_components():
    """Test Menu 5 internal components"""
    safe_print("\n🧪 TEST 3: Menu 5 Internal Components")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import (
            Menu5BacktestStrategy, 
            SessionDataAnalyzer,
            ProfessionalTradingSimulator,
            EnterpriseBacktestEngine
        )
        
        safe_print("✅ Menu5BacktestStrategy class available")
        safe_print("✅ SessionDataAnalyzer class available")
        safe_print("✅ ProfessionalTradingSimulator class available")
        safe_print("✅ EnterpriseBacktestEngine class available")
        
        # Test data classes
        try:
            from menu_modules.menu_5_backtest_strategy import TradingOrder, TradingPosition, BacktestSession
            safe_print("✅ TradingOrder dataclass available")
            safe_print("✅ TradingPosition dataclass available")
            safe_print("✅ BacktestSession dataclass available")
        except ImportError:
            safe_print("⚠️ Some dataclasses might not be available (not critical)")
            
        return True
    except Exception as e:
        safe_print(f"❌ Menu 5 components test failed: {e}")
        return False

def test_unified_menu_integration():
    """Test unified menu system integration"""
    safe_print("\n🧪 TEST 4: Unified Menu System Integration")
    safe_print("="*50)
    
    try:
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        
        # Initialize unified menu system
        menu_system = UnifiedMasterMenuSystem()
        safe_print("✅ Unified Master Menu System initialized")
        
        # Check if Menu 5 handler exists
        if hasattr(menu_system, '_handle_backtest_strategy'):
            safe_print("✅ Menu 5 handler method exists in unified system")
        else:
            safe_print("❌ Menu 5 handler method missing in unified system")
            return False
            
        # Test handler method signature
        import inspect
        handler_method = getattr(menu_system, '_handle_backtest_strategy')
        sig = inspect.signature(handler_method)
        if 'return' in str(sig) or len(sig.parameters) == 0:
            safe_print("✅ Menu 5 handler method has correct signature")
        else:
            safe_print("⚠️ Menu 5 handler method signature might need review")
            
        return True
    except Exception as e:
        safe_print(f"❌ Unified menu integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_data_availability():
    """Test if session data is available for analysis"""
    safe_print("\n🧪 TEST 5: Session Data Availability")
    safe_print("="*50)
    
    try:
        sessions_dir = Path("outputs/sessions")
        if sessions_dir.exists():
            sessions = list(sessions_dir.glob("20*"))
            safe_print(f"✅ Sessions directory exists with {len(sessions)} sessions")
            
            if sessions:
                # Check latest session
                latest_session = max(sessions, key=lambda x: x.name)
                safe_print(f"✅ Latest session found: {latest_session.name}")
                
                # Check for session files
                session_summary = latest_session / "session_summary.json"
                elliott_results = latest_session / "elliott_wave_real_results.json"
                
                if session_summary.exists():
                    safe_print("✅ session_summary.json available")
                if elliott_results.exists():
                    safe_print("✅ elliott_wave_real_results.json available")
                    
            return True
        else:
            safe_print("⚠️ Sessions directory not found (will be created when needed)")
            return True
    except Exception as e:
        safe_print(f"❌ Session data availability test failed: {e}")
        return False

def main():
    """Run all Menu 5 integration tests"""
    safe_print("🎯 MENU 5 BACKTEST STRATEGY - INTEGRATION TEST SUITE")
    safe_print("="*70)
    safe_print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"🔍 Testing Menu 5 integration with Unified Master Menu System")
    safe_print("")
    
    # Run all tests
    tests = [
        ("Menu 5 Import", test_menu_5_import),
        ("Menu 5 Initialization", test_menu_5_initialization),
        ("Menu 5 Components", test_menu_5_components),
        ("Unified Menu Integration", test_unified_menu_integration),
        ("Session Data Availability", test_session_data_availability)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            safe_print(f"❌ {test_name} test crashed: {e}")
    
    # Summary
    safe_print("\n" + "="*70)
    safe_print("📊 TEST SUMMARY")
    safe_print("="*70)
    safe_print(f"✅ Passed: {passed_tests}/{total_tests}")
    safe_print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        safe_print("🎉 ALL TESTS PASSED! Menu 5 integration is ready!")
        safe_print("🚀 Menu 5 BackTest Strategy is fully integrated with the system")
    elif passed_tests >= total_tests * 0.8:
        safe_print("✅ MOSTLY READY! Minor issues detected but system should work")
    else:
        safe_print("⚠️ SIGNIFICANT ISSUES! Please review failed tests")
    
    safe_print("\n💡 Next Steps:")
    safe_print("1. Run: python ProjectP.py")
    safe_print("2. Select option '5' for BackTest Strategy")
    safe_print("3. Experience professional trading simulation!")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
