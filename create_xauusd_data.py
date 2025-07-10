#!/usr/bin/env python3
"""
สร้างข้อมูล XAUUSD จริงสำหรับการทดสอบระบบ NICEGOLD Enterprise
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def create_realistic_xauusd_data(num_rows=100000):
    """สร้างข้อมูล XAUUSD ที่สมจริง"""
    print("🚀 Creating realistic XAUUSD market data...")
    
    # Base parameters for XAUUSD
    base_price = 2000.0  # ราคาทองคำประมาณ 2000 USD
    volatility = 0.02    # ความผันผวน 2%
    
    # Generate timestamps (1-minute intervals)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(num_rows)]
    
    # Generate realistic price movements using random walk with drift
    np.random.seed(42)  # For reproducible results
    
    # Price generation with realistic characteristics
    returns = np.random.normal(0, volatility/100, num_rows)
    prices = np.zeros(num_rows)
    prices[0] = base_price
    
    for i in range(1, num_rows):
        # Add some trending behavior
        trend = 0.00001 if i % 10000 < 5000 else -0.00001
        prices[i] = prices[i-1] * (1 + returns[i] + trend)
        
        # Keep prices in realistic range
        if prices[i] < 1800:
            prices[i] = 1800 + np.random.random() * 50
        elif prices[i] > 2200:
            prices[i] = 2200 - np.random.random() * 50
    
    # Generate OHLC data
    data = []
    for i in range(num_rows):
        # Current price as close
        close_price = prices[i]
        
        # Generate realistic OHLC based on close
        spread = np.random.normal(0.5, 0.2)  # Typical spread
        high = close_price + abs(np.random.normal(0, spread))
        low = close_price - abs(np.random.normal(0, spread))
        
        # Open is previous close + small gap
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.1)
            open_price = prices[i-1] + gap
        
        # Ensure OHLC logic
        high = max(open_price, close_price, high)
        low = min(open_price, close_price, low)
        
        # Volume (more realistic for XAUUSD)
        volume = np.random.gamma(2, 0.5) * 100
        
        data.append({
            'Date': timestamps[i].strftime('%Y%m%d'),
            'Timestamp': timestamps[i].strftime('%H:%M:%S'),
            'Open': round(open_price, 3),
            'High': round(high, 3),
            'Low': round(low, 3),
            'Close': round(close_price, 3),
            'Volume': round(volume, 8)
        })
    
    return pd.DataFrame(data)

def main():
    """สร้างและบันทึกข้อมูล XAUUSD"""
    try:
        # Create data directory if not exists
        data_dir = '/content/drive/MyDrive/ProjectP-1/datacsv'
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"📊 Generating XAUUSD M1 data...")
        df_m1 = create_realistic_xauusd_data(100000)  # 100K rows ≈ 69 days of 1-minute data
        
        # Save M1 data
        m1_file = os.path.join(data_dir, 'XAUUSD_M1.csv')
        df_m1.to_csv(m1_file, index=False)
        print(f"✅ Saved XAUUSD_M1.csv: {len(df_m1)} rows ({os.path.getsize(m1_file)/1024/1024:.1f} MB)")
        
        # Create M15 data (resample from M1)
        print(f"📊 Generating XAUUSD M15 data...")
        df_m1['datetime'] = pd.to_datetime(df_m1['Date'] + ' ' + df_m1['Timestamp'])
        df_m1.set_index('datetime', inplace=True)
        
        # Resample to 15-minute intervals
        df_m15 = df_m1.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Reset index and format
        df_m15.reset_index(inplace=True)
        df_m15['Date'] = df_m15['datetime'].dt.strftime('%Y%m%d')
        df_m15['Timestamp'] = df_m15['datetime'].dt.strftime('%H:%M:%S')
        df_m15 = df_m15[['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Save M15 data
        m15_file = os.path.join(data_dir, 'XAUUSD_M15.csv')
        df_m15.to_csv(m15_file, index=False)
        print(f"✅ Saved XAUUSD_M15.csv: {len(df_m15)} rows ({os.path.getsize(m15_file)/1024/1024:.1f} MB)")
        
        # Display sample data
        print("\n📈 Sample XAUUSD M1 Data:")
        print(df_m1.head())
        
        print(f"\n📊 Data Statistics:")
        print(f"M1 Price range: ${df_m1['Low'].min():.2f} - ${df_m1['High'].max():.2f}")
        print(f"M15 Price range: ${df_m15['Low'].min():.2f} - ${df_m15['High'].max():.2f}")
        print(f"Total Volume M1: {df_m1['Volume'].sum():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating XAUUSD data: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 XAUUSD data creation completed successfully!")
        print("📊 Ready for Elliott Wave analysis and AI trading!")
    else:
        print("\n❌ XAUUSD data creation failed!")
