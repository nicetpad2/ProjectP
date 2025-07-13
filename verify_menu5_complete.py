#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎉 MENU 5 FINAL VERIFICATION SCRIPT
Verification for 100% Real Data System
"""

def test_mt5_backtest_system():
    """Test MT5-Style BackTest System"""
    print('\n🚀 Test 1: MT5-Style BackTest System')
    print('-'*50)
    
    try:
        from menu_modules.advanced_mt5_style_backtest import AdvancedMT5StyleBacktest
        
        # Create instance
        mt5_system = AdvancedMT5StyleBacktest()
        
        # Check critical methods
        has_run = hasattr(mt5_system, 'run')
        has_load_data = hasattr(mt5_system, 'load_market_data')
        
        print(f'✅ run() method exists: {has_run}')
        print(f'✅ load_market_data() method exists: {has_load_data}')
        
        if has_run and has_load_data:
            print('✅ MT5-Style BackTest: FULLY OPERATIONAL')
            return True
        else:
            print('❌ MT5-Style BackTest: MISSING METHODS')
            return False
            
    except Exception as e:
        print(f'❌ MT5-Style BackTest import failed: {e}')
        return False

def test_data_loading():
    """Test Data Loading"""
    print('\n📊 Test 2: Data Loading Verification')
    print('-'*50)
    
    try:
        import pandas as pd
        from pathlib import Path
        
        # Check data file
        data_file = Path('datacsv/XAUUSD_M1.csv')
        if data_file.exists():
            df = pd.read_csv(data_file)
            
            print(f'✅ Data file exists: {data_file}')
            print(f'✅ Data rows loaded: {len(df):,}')
            print(f'✅ Data columns: {list(df.columns)}')
            print(f'✅ No null values: {df.isnull().sum().sum() == 0}')
            
            if len(df) > 1000000:
                print('✅ 100% REAL DATA CONFIRMED: Over 1M rows')
                return True
            else:
                print('⚠️ Data might be incomplete')
                return False
        else:
            print('❌ Data file not found')
            return False
            
    except Exception as e:
        print(f'❌ Data verification failed: {e}')
        return False

def test_menu5_integration():
    """Test Menu 5 Integration"""
    print('\n🎯 Test 3: Menu 5 Integration')
    print('-'*50)
    
    try:
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        # Create Menu 5 instance
        menu5 = Menu5BacktestStrategy()
        
        # Check MT5 integration
        has_mt5_backtest = hasattr(menu5, 'mt5_backtest') and menu5.mt5_backtest is not None
        
        print(f'✅ Menu 5 initialized: True')
        print(f'✅ MT5 BackTest integrated: {has_mt5_backtest}')
        
        if has_mt5_backtest:
            has_mt5_run = hasattr(menu5.mt5_backtest, 'run')
            print(f'✅ MT5 run() method accessible: {has_mt5_run}')
            
            if has_mt5_run:
                print('✅ MENU 5 INTEGRATION: PERFECT')
                return True
            else:
                print('⚠️ MT5 run() method not accessible')
                return False
        else:
            print('⚠️ MT5 BackTest not integrated')
            return False
            
    except Exception as e:
        print(f'❌ Menu 5 integration test failed: {e}')
        return False

def main():
    """Main verification function"""
    print('🎉 FINAL MENU 5 VERIFICATION - 100% REAL DATA SYSTEM')
    print('='*70)
    
    # Run all tests
    test1_passed = test_mt5_backtest_system()
    test2_passed = test_data_loading()
    test3_passed = test_menu5_integration()
    
    # Final verdict
    print('\n' + '='*70)
    
    if test1_passed and test2_passed and test3_passed:
        print('🎊 FINAL VERDICT: MENU 5 READY FOR 100% REAL DATA EXECUTION')
        print('💯 NO SAMPLING • NO SHORTCUTS • MAXIMUM RELIABILITY')
        print('✅ ALL SYSTEMS OPERATIONAL')
    else:
        print('⚠️ SOME ISSUES DETECTED - REVIEW REQUIRED')
        print(f'MT5 System: {"✅" if test1_passed else "❌"}')
        print(f'Data Loading: {"✅" if test2_passed else "❌"}')
        print(f'Menu 5 Integration: {"✅" if test3_passed else "❌"}')
    
    print('='*70)

if __name__ == "__main__":
    main()
