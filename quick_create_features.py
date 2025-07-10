#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ Quick Feature Creation Script
Fast generation of features for testing and development
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def quick_create_features():
    """⚡ Quick feature creation for testing"""
    
    print("⚡ Quick Feature Creation Script")
    print("=" * 40)
    
    # Create sample data
    n_samples = 1000
    
    # Generate basic features
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1T'),
        'open': np.random.normal(2000, 10, n_samples),
        'high': np.random.normal(2005, 12, n_samples),
        'low': np.random.normal(1995, 8, n_samples),
        'close': np.random.normal(2000, 10, n_samples),
        'volume': np.random.randint(100, 1000, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'wave_count': np.random.randint(1, 6, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.52, 0.48])
    }
    
    df = pd.DataFrame(data)
    
    # Create datacsv directory if it doesn't exist
    os.makedirs('datacsv', exist_ok=True)
    
    # Save to CSV
    df.to_csv('datacsv/quick_features.csv', index=False)
    
    print(f"✅ Created {len(df)} sample records")
    print(f"📊 Features: {len(df.columns)} columns") 
    print(f"💾 Saved to: datacsv/quick_features.csv")
    
    return df

if __name__ == "__main__":
    df = quick_create_features()
    print("\n📋 Sample data:")
    print(df.head())
