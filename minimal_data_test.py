#!/usr/bin/env python3
"""
🔬 MINIMAL DATA LOADING TEST
Test data loading without heavy dependencies
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("🔬 MINIMAL DATA LOADING TEST")
print("=" * 50)

try:
    # Test 1: Can we import pandas?
    print("📦 Testing pandas import...")
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not available - using basic file reading")
        pd = None
    
    # Test 2: Check data files
    print("\n📁 Testing data file access...")
    datacsv_path = Path("datacsv")
    
    if not datacsv_path.exists():
        print(f"❌ datacsv directory not found: {datacsv_path.absolute()}")
        sys.exit(1)
    
    csv_files = list(datacsv_path.glob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files")
    
    # Use largest file
    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"📈 Using: {largest_file.name}")
    
    # Test 3: Basic file reading
    print(f"\n📊 Testing basic file reading...")
    
    if pd is not None:
        # Test with pandas
        print("Using pandas to read data...")
        try:
            # Read small sample first
            sample_df = pd.read_csv(largest_file, nrows=1000)
            print(f"✅ Sample read: {len(sample_df)} rows, {len(sample_df.columns)} columns")
            print(f"📋 Columns: {list(sample_df.columns)}")
            
            # Try to read full file
            print("Reading full file...")
            full_df = pd.read_csv(largest_file)
            print(f"✅ Full file read: {len(full_df):,} rows")
            
            # Check for issues that could cause data loss
            print(f"\n🔍 Data quality check...")
            
            # Missing values
            missing = full_df.isnull().sum()
            total_missing = missing.sum()
            print(f"❓ Total missing values: {total_missing:,}")
            
            # Check OHLC columns
            ohlc_cols = [col for col in full_df.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])]
            print(f"📈 OHLC columns found: {ohlc_cols}")
            
            # Check for problematic rows
            if 'Close' in full_df.columns:
                close_col = 'Close'
            elif 'close' in full_df.columns:
                close_col = 'close'
            else:
                close_col = None
            
            if close_col:
                # Check for zero or negative prices
                zero_prices = (full_df[close_col] <= 0).sum()
                print(f"💰 Zero/negative prices: {zero_prices}")
                
                # Check for extreme outliers
                mean_price = full_df[close_col].mean()
                std_price = full_df[close_col].std()
                outliers = ((full_df[close_col] < mean_price - 5*std_price) | 
                           (full_df[close_col] > mean_price + 5*std_price)).sum()
                print(f"📊 Extreme outliers: {outliers}")
            
            # Simulate dropna operation that might be causing issues
            print(f"\n🧪 Simulating dropna operations...")
            before_dropna = len(full_df)
            
            # Test different dropna scenarios
            after_any_dropna = len(full_df.dropna())
            after_subset_dropna = len(full_df.dropna(subset=ohlc_cols)) if ohlc_cols else before_dropna
            
            print(f"📊 Before any dropna: {before_dropna:,}")
            print(f"📊 After dropna() [any]: {after_any_dropna:,} ({100*after_any_dropna/before_dropna:.1f}% retained)")
            print(f"📊 After dropna(subset=OHLC): {after_subset_dropna:,} ({100*after_subset_dropna/before_dropna:.1f}% retained)")
            
            if after_any_dropna < before_dropna * 0.1:
                print("🚨 CRITICAL: dropna() removes >90% of data!")
            elif after_any_dropna < before_dropna * 0.5:
                print("⚠️ WARNING: dropna() removes >50% of data!")
            else:
                print("✅ dropna() impact is reasonable")
            
        except Exception as e:
            print(f"❌ pandas reading failed: {e}")
            pd = None
    
    if pd is None:
        # Fallback to basic file reading
        print("Using basic file reading...")
        with open(largest_file, 'r') as f:
            lines = [f.readline() for _ in range(1000)]
        
        print(f"✅ Basic read: {len(lines)} lines")
        print(f"📋 Header: {lines[0].strip()}")
    
    print(f"\n" + "=" * 50)
    print("🏁 MINIMAL TEST COMPLETE")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
