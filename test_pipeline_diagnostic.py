#!/usr/bin/env python3
"""
🔍 PIPELINE DATA LOSS DIAGNOSTIC
Track exactly where data is being lost in the pipeline
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings and CUDA
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_pipeline():
    """Test each stage of the data pipeline"""
    print("🔍 PIPELINE DATA LOSS DIAGNOSTIC")
    print("=" * 60)
    
    try:
        print("📦 Importing modules...")
        
        # Try to import required modules
        try:
            import pandas as pd
            import numpy as np
            print("✅ pandas and numpy imported")
        except ImportError as e:
            print(f"❌ Cannot import pandas/numpy: {e}")
            return
        
        try:
            from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
            print("✅ ElliottWaveDataProcessor imported")
        except ImportError as e:
            print(f"❌ Cannot import ElliottWaveDataProcessor: {e}")
            return
        
        # Step 1: Test raw data loading
        print(f"\n📊 Step 1: Testing raw data loading...")
        processor = ElliottWaveDataProcessor()
        
        try:
            raw_data = processor.load_real_data()
            if raw_data is not None:
                print(f"✅ Raw data loaded: {len(raw_data):,} rows, {len(raw_data.columns)} columns")
                print(f"📋 Columns: {list(raw_data.columns)}")
                
                # Check data types
                print(f"🔢 Data types:")
                for col, dtype in raw_data.dtypes.items():
                    print(f"   {col}: {dtype}")
                
                # Check for missing values
                missing = raw_data.isnull().sum()
                total_missing = missing.sum()
                print(f"❓ Missing values: {total_missing:,} total")
                if total_missing > 0:
                    print("   Missing by column:")
                    for col, miss in missing.items():
                        if miss > 0:
                            print(f"     {col}: {miss:,}")
            else:
                print("❌ Raw data is None!")
                return
        except Exception as e:
            print(f"❌ Raw data loading failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 2: Test feature engineering
        print(f"\n⚙️ Step 2: Testing feature engineering...")
        try:
            features = processor.create_elliott_wave_features(raw_data)
            if features is not None:
                print(f"✅ Features created: {len(features):,} rows, {len(features.columns)} features")
                
                # Calculate data loss
                if len(features) < len(raw_data):
                    loss_pct = (1 - len(features)/len(raw_data)) * 100
                    print(f"⚠️ Data loss in feature engineering: {loss_pct:.1f}%")
                    print(f"   Before: {len(raw_data):,} rows")
                    print(f"   After:  {len(features):,} rows")
                    print(f"   Lost:   {len(raw_data) - len(features):,} rows")
                else:
                    print("✅ No data loss in feature engineering")
                
                # Check for missing values in features
                missing_features = features.isnull().sum()
                total_missing_features = missing_features.sum()
                print(f"❓ Missing values in features: {total_missing_features:,} total")
                if total_missing_features > 0:
                    high_missing = missing_features[missing_features > len(features) * 0.1]
                    if len(high_missing) > 0:
                        print("   Features with >10% missing:")
                        for col, miss in high_missing.items():
                            print(f"     {col}: {miss:,} ({miss/len(features)*100:.1f}%)")
            else:
                print("❌ Feature engineering returned None!")
                return
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 3: Test ML data preparation
        print(f"\n🤖 Step 3: Testing ML data preparation...")
        try:
            X, y = processor.prepare_ml_data(features)
            if X is not None and y is not None:
                print(f"✅ ML data prepared:")
                print(f"   X shape: {X.shape}")
                print(f"   y shape: {y.shape}")
                print(f"   ML rows: {len(X):,}")
                
                # Calculate data loss from features to ML data
                if len(X) < len(features):
                    loss_pct = (1 - len(X)/len(features)) * 100
                    print(f"⚠️ Data loss in ML preparation: {loss_pct:.1f}%")
                    print(f"   Before: {len(features):,} rows")
                    print(f"   After:  {len(X):,} rows")
                    print(f"   Lost:   {len(features) - len(X):,} rows")
                else:
                    print("✅ No data loss in ML preparation")
                
                # Check target distribution
                if hasattr(y, 'value_counts'):
                    target_dist = y.value_counts()
                    print(f"📊 Target distribution:")
                    for val, count in target_dist.items():
                        print(f"   {val}: {count:,} ({count/len(y)*100:.1f}%)")
                
                # Overall pipeline loss
                total_loss_pct = (1 - len(X)/len(raw_data)) * 100
                print(f"\n📈 OVERALL PIPELINE SUMMARY:")
                print(f"   Original data: {len(raw_data):,} rows")
                print(f"   Final ML data: {len(X):,} rows")
                print(f"   Total loss:    {total_loss_pct:.1f}%")
                print(f"   Retained:      {100-total_loss_pct:.1f}%")
                
                if len(X) < 1000:
                    print("🚨 CRITICAL: Insufficient data for ML training!")
                elif len(X) < 10000:
                    print("⚠️ WARNING: Low data count may affect model performance")
                else:
                    print("✅ Sufficient data for ML training")
                    
            else:
                print("❌ ML data preparation returned None!")
                return
                
        except Exception as e:
            print(f"❌ ML data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n" + "=" * 60)
        print("🏁 PIPELINE DIAGNOSTIC COMPLETE")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_pipeline()
