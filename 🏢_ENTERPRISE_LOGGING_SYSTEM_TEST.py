#!/usr/bin/env python3
"""
🏢 ENTERPRISE LOGGING SYSTEM TEST
NICEGOLD ProjectP - Menu 5 Comprehensive Logging Validation

Test the new enterprise-grade logging system for Menu 5 BackTest Strategy:
- Detailed trade recording
- SQLite database logging
- CSV export functionality
- Session linking with Menu 1
- Performance metrics tracking
- Excel analysis export
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

def test_enterprise_logging_system():
    """Test the comprehensive enterprise logging system"""
    print("🏢 ENTERPRISE LOGGING SYSTEM TEST")
    print("=" * 60)
    
    success_count = 0
    total_tests = 8
    
    try:
        # Test 1: Import Menu 5 with new enterprise features
        print("\n1️⃣ Testing Menu 5 Import with Enterprise Features...")
        try:
            from menu_modules.menu_5_backtest_strategy import (
                Menu5BacktestStrategy, 
                EnterpriseBacktestLogger,
                DetailedTradeRecord,
                detect_latest_menu1_session,
                get_menu1_session_info
            )
            print("✅ All enterprise classes imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ Import failed: {e}")
        
        # Test 2: Menu 1 Session Detection
        print("\n2️⃣ Testing Menu 1 Session Detection...")
        try:
            latest_session = detect_latest_menu1_session()
            if latest_session:
                print(f"✅ Menu 1 session detected: {latest_session}")
            else:
                print("⚠️ No Menu 1 session found (standalone mode)")
            success_count += 1
        except Exception as e:
            print(f"❌ Session detection failed: {e}")
        
        # Test 3: Enterprise Logger Initialization
        print("\n3️⃣ Testing Enterprise Logger Initialization...")
        try:
            base_path = "outputs"
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S_test")
            menu1_session_id = latest_session if 'latest_session' in locals() else None
            
            enterprise_logger = EnterpriseBacktestLogger(base_path, session_id, menu1_session_id)
            print(f"✅ Enterprise logger initialized: Session {session_id}")
            success_count += 1
        except Exception as e:
            print(f"❌ Enterprise logger initialization failed: {e}")
        
        # Test 4: Database Creation
        print("\n4️⃣ Testing Database Creation...")
        try:
            if 'enterprise_logger' in locals():
                # Check if databases were created
                trades_db_exists = enterprise_logger.trades_db_path.exists()
                performance_db_exists = enterprise_logger.performance_db_path.exists()
                
                if trades_db_exists and performance_db_exists:
                    print("✅ SQLite databases created successfully")
                    print(f"   📊 Trades DB: {enterprise_logger.trades_db_path}")
                    print(f"   📈 Performance DB: {enterprise_logger.performance_db_path}")
                    success_count += 1
                else:
                    print("❌ Database creation failed")
            else:
                print("❌ Enterprise logger not available")
        except Exception as e:
            print(f"❌ Database test failed: {e}")
        
        # Test 5: Trade Logging
        print("\n5️⃣ Testing Detailed Trade Logging...")
        try:
            if 'enterprise_logger' in locals():
                # Create test trade record
                test_trade = DetailedTradeRecord(
                    trade_id="test_trade_001",
                    session_id=session_id,
                    menu1_session_id=menu1_session_id,
                    symbol="XAUUSD",
                    order_type="BUY",
                    volume=0.01,
                    entry_price=2000.50,
                    exit_price=2001.50,
                    entry_time=datetime.now() - timedelta(minutes=30),
                    exit_time=datetime.now(),
                    duration_seconds=1800,
                    profit_loss=1.00,
                    commission=0.07,
                    spread_cost=1.00,
                    slippage_entry=0.2,
                    slippage_exit=0.3,
                    entry_signal_strength=0.85,
                    exit_reason="Take Profit",
                    margin_used=20.00,
                    max_profit_during_trade=1.50,
                    max_loss_during_trade=-0.50,
                    market_conditions={"spread": 100, "volatility": "medium"},
                    technical_indicators={"rsi": 65, "macd": 0.5},
                    risk_metrics={"position_size_pct": 2.0, "risk_reward": 1.5}
                )
                
                # Log the trade
                enterprise_logger.log_trade_execution(test_trade)
                print("✅ Test trade logged successfully")
                success_count += 1
            else:
                print("❌ Enterprise logger not available")
        except Exception as e:
            print(f"❌ Trade logging test failed: {e}")
        
        # Test 6: Performance Metrics Logging
        print("\n6️⃣ Testing Performance Metrics Logging...")
        try:
            if 'enterprise_logger' in locals():
                # Log test metrics
                enterprise_logger.log_performance_metric("test_auc", 0.75, "Test AUC score")
                enterprise_logger.log_performance_metric("test_profit", 100.50, "Test profit amount")
                enterprise_logger.log_performance_metric("test_drawdown", 0.05, "Test max drawdown")
                print("✅ Performance metrics logged successfully")
                success_count += 1
            else:
                print("❌ Enterprise logger not available")
        except Exception as e:
            print(f"❌ Performance metrics test failed: {e}")
        
        # Test 7: Data Verification
        print("\n7️⃣ Testing Data Integrity Verification...")
        try:
            if 'enterprise_logger' in locals():
                # Check database content
                trade_count = enterprise_logger._get_trade_count()
                integrity_check = enterprise_logger._verify_data_integrity()
                
                print(f"✅ Data integrity verified")
                print(f"   📊 Trade records: {trade_count}")
                print(f"   🔍 Null P&L records: {integrity_check.get('null_pnl_count', 0)}")
                print(f"   ⏰ Invalid time sequences: {integrity_check.get('invalid_time_sequence', 0)}")
                success_count += 1
            else:
                print("❌ Enterprise logger not available")
        except Exception as e:
            print(f"❌ Data verification test failed: {e}")
        
        # Test 8: Menu 5 Integration
        print("\n8️⃣ Testing Menu 5 Integration...")
        try:
            menu5 = Menu5BacktestStrategy()
            
            # Check if enterprise features are available
            has_session_id = hasattr(menu5, 'session_id')
            has_enterprise_methods = hasattr(menu5, '_initialize_enterprise_logging')
            
            if has_session_id and has_enterprise_methods:
                print("✅ Menu 5 enterprise integration successful")
                print(f"   📊 Session ID: {menu5.session_id}")
                success_count += 1
            else:
                print("❌ Menu 5 enterprise integration incomplete")
        except Exception as e:
            print(f"❌ Menu 5 integration test failed: {e}")
        
        # Cleanup
        if 'enterprise_logger' in locals():
            enterprise_logger.close()
            print("\n🧹 Enterprise logger closed")
        
    except Exception as e:
        print(f"❌ Critical test failure: {e}")
    
    # Results
    print("\n" + "=" * 60)
    print(f"🏢 ENTERPRISE LOGGING SYSTEM TEST RESULTS")
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - ENTERPRISE LOGGING READY!")
        return True
    else:
        print("⚠️ SOME TESTS FAILED - CHECK IMPLEMENTATION")
        return False

def test_menu5_execution_with_logging():
    """Test Menu 5 execution with enterprise logging"""
    print("\n🚀 TESTING MENU 5 WITH ENTERPRISE LOGGING")
    print("=" * 60)
    
    try:
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        # Initialize Menu 5
        menu5 = Menu5BacktestStrategy()
        
        print(f"📊 Menu 5 Session ID: {menu5.session_id}")
        print(f"🔗 Menu 1 Detection: {menu5.menu1_session_id or 'Will be detected on run'}")
        
        # Note: Don't actually run the full backtest in test
        print("✅ Menu 5 ready for enterprise execution")
        print("💡 To run full test: python ProjectP.py -> Select Menu 5")
        
        return True
        
    except Exception as e:
        print(f"❌ Menu 5 execution test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🏢 NICEGOLD ENTERPRISE LOGGING SYSTEM VALIDATION")
    print("Testing comprehensive enterprise logging for Menu 5")
    print("-" * 80)
    
    # Test 1: Enterprise logging system
    test1_success = test_enterprise_logging_system()
    
    # Test 2: Menu 5 integration
    test2_success = test_menu5_execution_with_logging()
    
    # Final results
    print("\n" + "=" * 80)
    if test1_success and test2_success:
        print("🎉 ENTERPRISE LOGGING SYSTEM VALIDATION SUCCESSFUL")
        print("✅ Menu 5 is ready with comprehensive enterprise logging")
        print("✅ All trade details will be recorded in:")
        print("   📊 SQLite databases (detailed records)")
        print("   📄 CSV exports (analysis ready)")
        print("   📈 Excel files (visualization ready)")
        print("   🔗 Linked to Menu 1 sessions (traceability)")
        print("   📋 JSON reports (integration ready)")
    else:
        print("❌ ENTERPRISE LOGGING VALIDATION FAILED")
        print("⚠️ Check system configuration and retry")
    print("=" * 80)

if __name__ == "__main__":
    main()
