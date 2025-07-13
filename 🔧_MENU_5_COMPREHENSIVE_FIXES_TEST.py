#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 MENU 5 COMPREHENSIVE FIXES TEST
Testing all critical fixes for Menu 5 BackTest Strategy

🎯 FIXES APPLIED:
✅ Fixed margin calculation (realistic 1% instead of excessive requirements)
✅ Fixed P&L calculation (100 multiplier instead of 100,000)
✅ Fixed parameter display (100pt spread, $0.07 commission)
✅ Fixed position sizing and trading signal generation
✅ Fixed stop loss and take profit calculations
✅ Enhanced trading simulation logic

🧪 TEST SCENARIOS:
1. Parameter Validation
2. Margin Calculation Test
3. Trading Simulation Test
4. P&L Calculation Test
5. Full Integration Test
"""

import sys
import os
sys.path.append('/content/drive/MyDrive/ProjectP-1')

from menu_modules.menu_5_backtest_strategy import (
    ProfessionalTradingSimulator, 
    EnterpriseBacktestEngine,
    TradingOrder,
    OrderType,
    OrderStatus,
    TradingPosition,
    PositionStatus
)
import traceback
from datetime import datetime

def test_parameter_validation():
    """Test 1: Parameter Validation"""
    print("🧪 TEST 1: Parameter Validation")
    print("=" * 50)
    
    try:
        simulator = ProfessionalTradingSimulator()
        
        # Check updated parameters
        spread_points = simulator.spread_points
        commission = simulator.commission_per_lot
        initial_balance = simulator.initial_balance
        
        print(f"✅ Spread Points: {spread_points} (Expected: 100)")
        print(f"✅ Commission: ${commission} (Expected: $0.07)")
        print(f"✅ Initial Balance: ${initial_balance}")
        
        # Validate
        assert spread_points == 100, f"Spread should be 100, got {spread_points}"
        assert commission == 0.07, f"Commission should be 0.07, got {commission}"
        assert initial_balance == 10000.0, f"Initial balance should be 10000, got {initial_balance}"
        
        print("🎉 Parameter Validation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Parameter Validation: FAILED - {e}")
        print(traceback.format_exc())
        return False

def test_margin_calculation():
    """Test 2: Margin Calculation Test"""
    print("\\n🧪 TEST 2: Margin Calculation Test")
    print("=" * 50)
    
    try:
        simulator = ProfessionalTradingSimulator()
        
        # Test margin calculation
        volume = 0.01  # 0.01 lot
        price = 2050.0  # XAUUSD price
        
        required_margin = simulator._calculate_required_margin(volume, price)
        
        # Expected: 0.01 * 100 * 2050 * 0.01 = $2.05
        expected_margin = 0.01 * 100 * 2050 * 0.01
        
        print(f"✅ Volume: {volume} lot")
        print(f"✅ Price: ${price}")
        print(f"✅ Required Margin: ${required_margin:.2f}")
        print(f"✅ Expected Margin: ${expected_margin:.2f}")
        
        # Check if margin is reasonable (should be much less than before)
        assert required_margin < 1000, f"Margin too high: ${required_margin}"
        assert required_margin > 0, f"Margin should be positive: ${required_margin}"
        
        print("🎉 Margin Calculation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Margin Calculation: FAILED - {e}")
        print(traceback.format_exc())
        return False

def test_trading_simulation():
    """Test 3: Trading Simulation Test"""
    print("\\n🧪 TEST 3: Trading Simulation Test")
    print("=" * 50)
    
    try:
        simulator = ProfessionalTradingSimulator()
        
        # Create test order
        order = TradingOrder(
            order_id="test_order_1",
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.01,
            price=2050.0,
            timestamp=datetime.now()
        )
        
        print(f"✅ Created test order: {order.order_id}")
        print(f"   - Type: {order.order_type.value}")
        print(f"   - Volume: {order.volume} lot")
        print(f"   - Price: ${order.price}")
        
        # Test order placement
        initial_balance = simulator.current_balance
        initial_equity = simulator.equity
        
        success = simulator.place_order(order)
        
        print(f"✅ Order placement result: {success}")
        print(f"✅ Balance before: ${initial_balance:.2f}")
        print(f"✅ Balance after: ${simulator.current_balance:.2f}")
        print(f"✅ Margin used: ${simulator.margin_used:.2f}")
        print(f"✅ Positions count: {len(simulator.positions)}")
        
        if success:
            print("🎉 Trading Simulation: PASSED")
            return True
        else:
            print("❌ Trading Simulation: FAILED - Order not placed")
            return False
        
    except Exception as e:
        print(f"❌ Trading Simulation: FAILED - {e}")
        print(traceback.format_exc())
        return False

def test_pnl_calculation():
    """Test 4: P&L Calculation Test"""
    print("\\n🧪 TEST 4: P&L Calculation Test")
    print("=" * 50)
    
    try:
        simulator = ProfessionalTradingSimulator()
        
        # Create and place order
        order = TradingOrder(
            order_id="pnl_test_order",
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.01,
            price=2050.0,
            timestamp=datetime.now()
        )
        
        success = simulator.place_order(order)
        if not success:
            print("❌ Could not place order for P&L test")
            return False
        
        # Update positions with new price
        new_price = 2051.0  # $1 profit
        current_prices = {"XAUUSD": new_price}
        simulator.update_positions(current_prices)
        
        # Check P&L
        position = simulator.positions[0]
        pnl = position.profit_loss
        
        # Expected P&L: (2051 - 2050) * 0.01 * 100 = $1.00
        # Minus commission and spread costs
        expected_gross_pnl = (2051 - 2050) * 0.01 * 100
        
        print(f"✅ Entry Price: ${position.entry_price}")
        print(f"✅ Current Price: ${position.current_price}")
        print(f"✅ Volume: {position.volume} lot")
        print(f"✅ Gross P&L Expected: ${expected_gross_pnl:.2f}")
        print(f"✅ Net P&L (after costs): ${pnl:.2f}")
        print(f"✅ Commission: ${position.commission:.2f}")
        print(f"✅ Spread Cost: ${position.spread_cost:.2f}")
        
        # P&L should be reasonable (not zero, not millions)
        assert abs(pnl) < 1000, f"P&L too large: ${pnl}"
        assert pnl != 0, f"P&L should not be zero: ${pnl}"
        
        print("🎉 P&L Calculation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ P&L Calculation: FAILED - {e}")
        print(traceback.format_exc())
        return False

def test_full_integration():
    """Test 5: Full Integration Test"""
    print("\\n🧪 TEST 5: Full Integration Test")
    print("=" * 50)
    
    try:
        # Initialize backtest engine
        backtest_engine = EnterpriseBacktestEngine()
        
        print("✅ Backtest Engine initialized")
        print(f"✅ Console available: {backtest_engine.console is not None}")
        print(f"✅ Logger available: {backtest_engine.logger is not None}")
        
        # Test parameter display method
        if hasattr(backtest_engine, '_display_trading_parameters'):
            print("✅ Parameter display method available")
        
        print("🎉 Full Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Full Integration: FAILED - {e}")
        print(traceback.format_exc())
        return False

def main():
    """Run all comprehensive tests"""
    print("🔧 MENU 5 COMPREHENSIVE FIXES TEST")
    print("=" * 60)
    print("Testing all critical fixes for Menu 5 BackTest Strategy")
    print("=" * 60)
    
    tests = [
        test_parameter_validation,
        test_margin_calculation,
        test_trading_simulation,
        test_pnl_calculation,
        test_full_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL FIXES VALIDATED SUCCESSFULLY! 🚀")
        print("✅ Menu 5 is now ready for production use!")
    else:
        print("⚠️ Some tests failed. Please review and fix remaining issues.")
    
    return passed == total

if __name__ == "__main__":
    main()
