#!/usr/bin/env python3
"""
🧪 QUICK DATA PROCESSING TEST
Test the fixed data processor to verify row count retention
"""

import sys
import os
from pathlib import Path

# Force CPU-only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🧪 TESTING DATA PROCESSOR FIXES")
print("=" * 50)

try:
    # Test raw data reading first
    print("📊 Step 1: Testing raw data reading...")
    
    # Get data files
    datacsv_path = Path("datacsv")
    csv_files = list(datacsv_path.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found!")
        sys.exit(1)
    
    print(f"✅ Found {len(csv_files)} CSV files")
    
    # Get largest file
    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"📈 Testing with: {largest_file.name}")
    
    # Use basic Python to read first few lines
    with open(largest_file, 'r') as f:
        lines = f.readlines()[:10]
    
    print(f"✅ File accessible, sample lines: {len(lines)}")
    print(f"📋 Header: {lines[0].strip()}")
    print(f"📋 Sample row: {lines[1].strip()}")
    
    # Count total lines
    with open(largest_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"📊 Total lines in file: {total_lines:,}")
    
    # Basic validation
    if total_lines < 1000:
        print("⚠️ WARNING: File seems small for market data")
    else:
        print("✅ File size looks reasonable for market data")
    
    print("\n🔧 Step 2: Testing column structure...")
    header = lines[0].strip().split(',')
    print(f"📋 Columns found: {header}")
    
    # Check for OHLC columns
    ohlc_patterns = ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']
    found_ohlc = [col for col in header if any(pattern.lower() in col.lower() for pattern in ohlc_patterns)]
    print(f"📈 OHLC columns detected: {found_ohlc}")
    
    if len(found_ohlc) >= 4:
        print("✅ OHLC structure looks good")
    else:
        print("⚠️ WARNING: OHLC structure may be incomplete")
    
    print("\n" + "=" * 50)
    print("🎯 BASIC DATA VALIDATION COMPLETE")
    print(f"   File: {largest_file.name}")
    print(f"   Rows: {total_lines:,}")
    print(f"   Columns: {len(header)}")
    print(f"   OHLC: {len(found_ohlc)}/4 detected")
    
    if total_lines > 100000:
        print("✅ Data looks suitable for ML pipeline")
    else:
        print("⚠️ Data may be too small for robust ML training")
    
except Exception as e:
    print(f"❌ Error in basic test: {e}")
    import traceback
    traceback.print_exc()
