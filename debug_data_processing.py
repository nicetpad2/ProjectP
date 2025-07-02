#!/usr/bin/env python3
"""
Debug script to check data processing pipeline
"""

import sys
import pandas as pd
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))

def debug_data_processing():
    print("🔍 DEBUGGING DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Check available files
    datacsv_path = Path('datacsv')
    csv_files = list(datacsv_path.glob('*.csv'))
    print(f"\n📁 Available CSV files:")
    for file in csv_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.1f} MB")
    
    # Test file selection logic
    print(f"\n🎯 File Selection Logic:")
    timeframe_priority = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
    
    selected_file = None
    for timeframe in timeframe_priority:
        for file_path in csv_files:
            if timeframe in file_path.name.upper():
                selected_file = file_path
                print(f"  ✅ Selected: {selected_file.name} (priority: {timeframe})")
                break
        if selected_file:
            break
    
    if not selected_file:
        selected_file = csv_files[0]
        print(f"  ⚠️ No priority match, using: {selected_file.name}")
    
    # Test data loading
    print(f"\n📊 Testing Data Loading from {selected_file.name}:")
    try:
        # Load small sample first
        df_sample = pd.read_csv(selected_file, nrows=1000)
        print(f"  Sample loaded: {len(df_sample)} rows")
        print(f"  Columns: {list(df_sample.columns)}")
        print(f"  Sample data:")
        print(df_sample.head(3).to_string())
        
        # Test date parsing
        print(f"\n📅 Testing Date Parsing:")
        if 'Date' in df_sample.columns and 'Timestamp' in df_sample.columns:
            print(f"  Date sample: {df_sample['Date'].iloc[0]}")
            print(f"  Timestamp sample: {df_sample['Timestamp'].iloc[0]}")
            
            # Test date conversion
            try:
                date_str = str(df_sample['Date'].iloc[0])
                if len(date_str) >= 6:
                    year = '20' + date_str[:2]
                    month = date_str[2:4]
                    day = date_str[4:6]
                    converted = f"{year}-{month}-{day} {df_sample['Timestamp'].iloc[0]}"
                    print(f"  Converted datetime: {converted}")
                    
                    # Test pandas conversion
                    test_datetime = pd.to_datetime(converted)
                    print(f"  ✅ Pandas datetime: {test_datetime}")
                else:
                    print(f"  ❌ Date format invalid: {date_str}")
            except Exception as e:
                print(f"  ❌ Date conversion error: {e}")
    
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return
    
    # Test full file loading (with limit for safety)
    print(f"\n📈 Testing Full File Processing:")
    try:
        if selected_file.name == 'XAUUSD_M1.csv':
            # M1 is very large, limit to 100k rows for testing
            df_full = pd.read_csv(selected_file, nrows=100000)
        else:
            # M15 is smaller, can load all
            df_full = pd.read_csv(selected_file)
        
        print(f"  Initial rows: {len(df_full)}")
        
        # Test date processing
        if 'Date' in df_full.columns and 'Timestamp' in df_full.columns:
            try:
                df_full['date_str'] = df_full['Date'].astype(str)
                df_full['date_str'] = '20' + df_full['date_str'].str[:2] + '-' + df_full['date_str'].str[2:4] + '-' + df_full['date_str'].str[4:6]
                df_full['timestamp'] = pd.to_datetime(df_full['date_str'] + ' ' + df_full['Timestamp'].astype(str))
                df_full = df_full.drop(columns=['Date', 'Timestamp', 'date_str'])
                print(f"  After date processing: {len(df_full)} rows")
            except Exception as e:
                print(f"  ❌ Date processing error: {e}")
                return
        
        # Test data cleaning
        try:
            # Sort by timestamp
            df_full = df_full.sort_values('timestamp').reset_index(drop=True)
            print(f"  After sorting: {len(df_full)} rows")
            
            # Remove duplicates
            df_full = df_full.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            print(f"  After deduplication: {len(df_full)} rows")
            
            # Handle missing values
            df_full = df_full.ffill().bfill()
            print(f"  After filling missing: {len(df_full)} rows")
            
            # Check for valid OHLC
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in df_full.columns]
            if available_cols:
                df_full = df_full.dropna(subset=available_cols)
                print(f"  After OHLC validation: {len(df_full)} rows")
            
            print(f"  ✅ Final cleaned data: {len(df_full)} rows")
            
        except Exception as e:
            print(f"  ❌ Data cleaning error: {e}")
    
    except Exception as e:
        print(f"  ❌ Full file processing error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Debug completed!")

if __name__ == "__main__":
    debug_data_processing()
