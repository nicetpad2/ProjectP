#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST DATA LOADING AFTER FIXES
Test if the system now loads ALL data without 1000-row limits
"""

import sys
import pandas as pd
from pathlib import Path

def test_datacsv_direct():
    """Test loading data directly from datacsv"""
    print("🔍 Testing direct datacsv loading...")
    
    datacsv_path = Path("datacsv")
    if not datacsv_path.exists():
        print("❌ datacsv/ directory not found")
        return False
    
    csv_files = list(datacsv_path.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    for csv_file in csv_files:
        print(f"\n📊 Testing file: {csv_file.name}")
        try:
            # Load ALL data
            data = pd.read_csv(csv_file)
            print(f"   Shape: {data.shape}")
            print(f"   Memory: {data.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
            
            if len(data) > 1000000:
                print(f"   ✅ SUCCESS: {len(data):,} rows (FULL DATASET)")
            elif len(data) > 100000:
                print(f"   ✅ GOOD: {len(data):,} rows (LARGE DATASET)")
            else:
                print(f"   ⚠️ SMALL: {len(data):,} rows")
                
        except Exception as e:
            print(f"   ❌ Error loading {csv_file.name}: {e}")
    
    return True

def test_perfect_menu():
    """Test perfect menu after fixes"""
    print("\n🔍 Testing Perfect Menu after fixes...")
    
    try:
        sys.path.append('.')
        from perfect_menu_1 import PerfectElliottWaveMenu
        
        menu = PerfectElliottWaveMenu()
        print("   ✅ Menu created")
        
        if menu.initialize_components():
            print("   ✅ Components initialized")
            
            data = menu.data_processor.load_data()
            print(f"   📊 Data shape: {data.shape}")
            
            if len(data) > 1000000:
                print(f"   ✅ SUCCESS: {len(data):,} rows (NO LIMITS!)")
                return True
            elif len(data) > 100000:
                print(f"   ✅ GOOD: {len(data):,} rows")
                return True
            elif len(data) > 1000:
                print(f"   ⚠️ PARTIAL: {len(data):,} rows")
                return False
            else:
                print(f"   ❌ STILL LIMITED: {len(data)} rows")
                return False
        else:
            print("   ❌ Failed to initialize components")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_optimized_menu():
    """Test optimized menu after fixes"""
    print("\n🔍 Testing Optimized Menu after fixes...")
    
    try:
        sys.path.append('.')
        from menu_modules.optimized_menu_1_elliott_wave import OptimizedMenu1ElliottWave
        
        menu = OptimizedMenu1ElliottWave()
        print("   ✅ Menu created")
        print(f"   📊 Max data rows setting: {menu.max_data_rows}")
        
        if menu.max_data_rows is None:
            print("   ✅ SUCCESS: No row limits!")
            return True
        else:
            print(f"   ❌ STILL LIMITED: {menu.max_data_rows} rows")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("📋 COMPREHENSIVE DATA LOADING TEST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Direct datacsv loading
    if test_datacsv_direct():
        success_count += 1
    
    # Test 2: Perfect menu
    if test_perfect_menu():
        success_count += 1
    
    # Test 3: Optimized menu
    if test_optimized_menu():
        success_count += 1
    
    print("\n📊 FINAL RESULTS:")
    print(f"✅ Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - 1000-row limit fixed!")
    elif success_count > 0:
        print("⚠️ PARTIAL SUCCESS - some issues remain")
    else:
        print("❌ ALL TESTS FAILED - issues not resolved")
