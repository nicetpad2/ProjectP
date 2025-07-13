#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 MENU 5 PARAMETERS VALIDATION TEST
Validate the updated trading parameters (100pt spread, $0.07 commission)

✅ Tests:
1. Parameter values validation
2. Calculation accuracy
3. Display consistency
4. Integration verification
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

def test_parameter_values():
    """Test that parameters are set to correct values"""
    safe_print("🧪 TEST 1: Parameter Values Validation")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import ProfessionalTradingSimulator
        
        # Initialize simulator
        simulator = ProfessionalTradingSimulator()
        
        # Check spread parameter
        if simulator.spread_points == 100:
            safe_print("✅ Spread: 100 points (CORRECT)")
        else:
            safe_print(f"❌ Spread: {simulator.spread_points} points (INCORRECT - should be 100)")
            return False
            
        # Check commission parameter
        if simulator.commission_per_lot == 0.07:
            safe_print("✅ Commission: $0.07 per 0.01 lot (CORRECT)")
        else:
            safe_print(f"❌ Commission: ${simulator.commission_per_lot} per 0.01 lot (INCORRECT - should be $0.07)")
            return False
            
        safe_print("✅ All parameter values are correct!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Parameter validation failed: {e}")
        return False

def test_calculation_accuracy():
    """Test calculation accuracy with new parameters"""
    safe_print("\n🧪 TEST 2: Calculation Accuracy")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import ProfessionalTradingSimulator
        
        simulator = ProfessionalTradingSimulator()
        
        # Test spread cost calculation
        test_volume = 0.01  # 0.01 lot
        spread_cost = simulator.calculate_spread_cost(test_volume)
        expected_spread_cost = (100 / 100000) * test_volume * 100000  # 100 points in price
        
        safe_print(f"🔍 Spread Cost Test:")
        safe_print(f"   Volume: {test_volume} lot")
        safe_print(f"   Calculated: {spread_cost:.5f}")
        safe_print(f"   Expected: {expected_spread_cost:.5f}")
        
        if abs(spread_cost - expected_spread_cost) < 0.00001:
            safe_print("✅ Spread cost calculation: CORRECT")
        else:
            safe_print("❌ Spread cost calculation: INCORRECT")
            return False
            
        # Test commission calculation
        commission = simulator.calculate_commission(test_volume)
        expected_commission = (test_volume / 0.01) * 0.07  # $0.07 per 0.01 lot
        
        safe_print(f"\n🔍 Commission Test:")
        safe_print(f"   Volume: {test_volume} lot")
        safe_print(f"   Calculated: ${commission:.2f}")
        safe_print(f"   Expected: ${expected_commission:.2f}")
        
        if abs(commission - expected_commission) < 0.01:
            safe_print("✅ Commission calculation: CORRECT")
        else:
            safe_print("❌ Commission calculation: INCORRECT")
            return False
            
        safe_print("✅ All calculations are accurate!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_consistency():
    """Test display consistency across system"""
    safe_print("\n🧪 TEST 3: Display Consistency")
    safe_print("="*50)
    
    try:
        # Test Menu 5 display
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        config = {'test_mode': True}
        menu_5 = Menu5BacktestStrategy(config)
        
        safe_print("✅ Menu 5 display components loaded")
        
        # Test unified menu display
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        
        unified_menu = UnifiedMasterMenuSystem()
        safe_print("✅ Unified menu system loaded")
        
        # Check if handler exists
        if hasattr(unified_menu, '_handle_backtest_strategy'):
            safe_print("✅ Menu 5 handler exists in unified system")
        else:
            safe_print("❌ Menu 5 handler missing")
            return False
            
        safe_print("✅ Display consistency validated!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Display consistency test failed: {e}")
        return False

def test_integration_verification():
    """Test full system integration"""
    safe_print("\n🧪 TEST 4: Integration Verification")
    safe_print("="*50)
    
    try:
        # Test complete import chain
        from menu_modules.menu_5_backtest_strategy import (
            Menu5BacktestStrategy,
            ProfessionalTradingSimulator, 
            SessionDataAnalyzer,
            EnterpriseBacktestEngine
        )
        
        safe_print("✅ All Menu 5 components imported successfully")
        
        # Test initialization with parameters
        config = {'test_mode': True}
        menu_5 = Menu5BacktestStrategy(config)
        
        # Verify simulator has correct parameters
        simulator = ProfessionalTradingSimulator(config)
        
        parameters_correct = (
            simulator.spread_points == 100 and 
            simulator.commission_per_lot == 0.07
        )
        
        if parameters_correct:
            safe_print("✅ Integration verification: All parameters correct")
        else:
            safe_print("❌ Integration verification: Parameter mismatch")
            return False
            
        safe_print("✅ Full system integration verified!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Integration verification failed: {e}")
        return False

def test_real_world_scenario():
    """Test realistic trading scenario"""
    safe_print("\n🧪 TEST 5: Real-World Scenario")
    safe_print("="*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import ProfessionalTradingSimulator
        
        simulator = ProfessionalTradingSimulator()
        
        # Simulate a 1.0 lot trade
        volume = 1.0
        entry_price = 2000.00  # Example XAUUSD price
        
        # Calculate costs
        spread_cost = simulator.calculate_spread_cost(volume)
        commission = simulator.calculate_commission(volume)
        total_cost = spread_cost + commission
        
        safe_print(f"🔍 Real Trading Scenario (1.0 lot @ $2000):")
        safe_print(f"   📊 Spread Cost: ${spread_cost:.2f}")
        safe_print(f"   💰 Commission: ${commission:.2f}")
        safe_print(f"   💎 Total Cost: ${total_cost:.2f}")
        
        # Verify costs are reasonable
        if spread_cost > 0 and commission > 0 and total_cost < 200:  # Updated limit for 100pt spread
            safe_print("✅ Real-world scenario: Costs are reasonable")
        else:
            safe_print("❌ Real-world scenario: Costs seem incorrect")
            return False
            
        # Calculate breakeven more accurately
        # For XAUUSD: breakeven in points = total_cost / (pip_value_per_lot * volume) * points_per_pip
        pip_value_per_lot = 100  # $100 per pip per 1.0 lot for XAUUSD
        points_per_pip = 100  # 100 points = 1 pip
        breakeven_points = (total_cost / (pip_value_per_lot * volume)) * points_per_pip
        safe_print(f"   📈 Breakeven: {breakeven_points:.1f} points")
        
        if breakeven_points > 0 and breakeven_points < 200:  # Reasonable breakeven
            safe_print("✅ Breakeven calculation: Reasonable")
        else:
            safe_print("❌ Breakeven calculation: Unreasonable")
            return False
            
        safe_print("✅ Real-world scenario validation successful!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Real-world scenario test failed: {e}")
        return False

def main():
    """Run all parameter validation tests"""
    safe_print("🎯 MENU 5 PARAMETERS VALIDATION - TEST SUITE")
    safe_print("="*70)
    safe_print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("🔍 Testing updated trading parameters: 100pt spread, $0.07 commission")
    safe_print("")
    
    # Run all tests
    tests = [
        ("Parameter Values", test_parameter_values),
        ("Calculation Accuracy", test_calculation_accuracy),
        ("Display Consistency", test_display_consistency),
        ("Integration Verification", test_integration_verification),
        ("Real-World Scenario", test_real_world_scenario)
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
    safe_print("📊 PARAMETER VALIDATION SUMMARY")
    safe_print("="*70)
    safe_print(f"✅ Passed: {passed_tests}/{total_tests}")
    safe_print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        safe_print("🎉 ALL TESTS PASSED! New parameters are working perfectly!")
        safe_print("🚀 Menu 5 is ready with updated trading conditions:")
        safe_print("   📊 Spread: 100 points (1.00 pips)")
        safe_print("   💰 Commission: $0.07 per 0.01 lot")
    elif passed_tests >= total_tests * 0.8:
        safe_print("✅ MOSTLY READY! Minor issues detected but system should work")
    else:
        safe_print("⚠️ SIGNIFICANT ISSUES! Please review failed tests")
    
    safe_print("\n💡 Ready to test:")
    safe_print("1. Run: python ProjectP.py")
    safe_print("2. Select option '5' for BackTest Strategy")
    safe_print("3. Experience updated trading simulation with new parameters!")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
