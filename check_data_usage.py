#!/usr/bin/env python3
"""
🔍 DATA USAGE CHECKER
ตรวจสอบการใช้งานข้อมูลใน datacsv/ ว่าใช้ครบทุกบรรทัดหรือไม่
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def check_csv_data():
    """ตรวจสอบข้อมูลใน CSV files"""
    print("=" * 60)
    print("🔍 CHECKING DATA USAGE IN DATACSV FOLDER")
    print("=" * 60)
    
    datacsv_path = Path("datacsv")
    
    if not datacsv_path.exists():
        print("❌ datacsv folder not found!")
        return
    
    csv_files = list(datacsv_path.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in datacsv!")
        return
    
    total_original_rows = 0
    
    for csv_file in csv_files:
        print(f"\n📁 File: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            rows = len(df)
            cols = len(df.columns)
            missing = df.isnull().sum().sum()
            
            print(f"   Rows: {rows:,}")
            print(f"   Columns: {cols}")
            print(f"   Missing values: {missing}")
            print(f"   Data types: {df.dtypes.to_dict()}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            total_original_rows += rows
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n📊 TOTAL ORIGINAL DATA: {total_original_rows:,} rows")
    return total_original_rows

def check_data_processor_usage():
    """ตรวจสอบการใช้งานข้อมูลผ่าน DataProcessor"""
    print("\n" + "=" * 60)
    print("⚙️ CHECKING DATA PROCESSOR USAGE")
    print("=" * 60)
    
    try:
        from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
        from core.logger import setup_enterprise_logger
        
        # Setup logger
        logger = setup_enterprise_logger()
        
        # Create processor
        processor = ElliottWaveDataProcessor(logger=logger)
        
        print("\n1️⃣ Loading real data...")
        data = processor.load_real_data()
        
        if data is None:
            print("❌ Failed to load data")
            return 0
        
        print(f"   After load_real_data(): {len(data):,} rows")
        print(f"   Columns: {len(data.columns)}")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        
        print("\n2️⃣ Creating Elliott Wave features...")
        features = processor.create_elliott_wave_features(data)
        print(f"   After feature engineering: {len(features):,} rows")
        print(f"   Features count: {len(features.columns)}")
        print(f"   Missing values: {features.isnull().sum().sum()}")
        
        print("\n3️⃣ Preparing ML data...")
        X, y = processor.prepare_ml_data(features)
        print(f"   Final X shape: {X.shape}")
        print(f"   Final y shape: {y.shape}")
        print(f"   Missing values in X: {X.isnull().sum().sum()}")
        print(f"   Missing values in y: {y.isnull().sum()}")
        
        return len(X)
        
    except Exception as e:
        print(f"❌ Error in data processor test: {e}")
        import traceback
        traceback.print_exc()
        return 0

def analyze_data_loss():
    """วิเคราะห์การสูญเสียข้อมูล"""
    print("\n" + "=" * 60)
    print("📉 DATA LOSS ANALYSIS")
    print("=" * 60)
    
    original_count = check_csv_data()
    processed_count = check_data_processor_usage()
    
    if original_count > 0 and processed_count > 0:
        loss_count = original_count - processed_count
        loss_percentage = (loss_count / original_count) * 100
        usage_percentage = (processed_count / original_count) * 100
        
        print(f"\n📊 SUMMARY:")
        print(f"   Original data: {original_count:,} rows")
        print(f"   Final ML data: {processed_count:,} rows")
        print(f"   Lost data: {loss_count:,} rows ({loss_percentage:.2f}%)")
        print(f"   Data utilization: {usage_percentage:.2f}%")
        
        if loss_percentage > 50:
            print("⚠️  WARNING: High data loss detected!")
        elif loss_percentage > 20:
            print("⚠️  WARNING: Moderate data loss detected")
        else:
            print("✅ Data utilization is acceptable")
            
        # Analyze reasons for data loss
        print(f"\n🔍 LIKELY REASONS FOR DATA LOSS:")
        print(f"   - Duplicate timestamp removal")
        print(f"   - Missing value cleanup (forward/backward fill)")
        print(f"   - Feature engineering window requirements")
        print(f"   - Target variable creation (future price shift)")
        print(f"   - NaN removal after calculations")
        
    return {
        'original_count': original_count,
        'processed_count': processed_count,
        'usage_percentage': (processed_count / original_count) * 100 if original_count > 0 else 0
    }

if __name__ == "__main__":
    results = analyze_data_loss()
    
    # Write results to file
    with open("data_usage_report.txt", "w") as f:
        f.write("DATA USAGE ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Original data rows: {results['original_count']:,}\n")
        f.write(f"Final ML data rows: {results['processed_count']:,}\n")
        f.write(f"Data utilization: {results['usage_percentage']:.2f}%\n")
        
    print(f"\n💾 Report saved to: data_usage_report.txt")
