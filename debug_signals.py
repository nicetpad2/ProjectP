#!/usr/bin/env python3
"""
🔍 Debug Trading Signals Analysis Script
สำหรับตรวจสอบสาเหตุที่ไม่มีกำไรในการทดสอบ Menu 5

รายละเอียดการตรวจสอบ:
1. การสร้าง Technical Indicators
2. การสร้าง Trading Signals
3. การคำนวณ Position Size
4. การคำนวณ Stop Loss และ Take Profit
5. การจำลองการเทรด
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the unified logger
import sys
import os
sys.path.append('/content/drive/MyDrive/ProjectP-1')

from core.unified_enterprise_logger import UnifiedEnterpriseLogger

# Initialize logger
logger = UnifiedEnterpriseLogger()

def debug_trading_signals():
    """Debug trading signals generation and execution"""
    try:
        logger.info("🔍 Starting Trading Signals Debug Analysis")
        
        # Load data
        data_file = "/content/drive/MyDrive/ProjectP-1/datacsv/XAUUSD_M1.csv"
        logger.info(f"📊 Loading data from: {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"✅ Data loaded: {len(df)} rows")
        
        # Create datetime column
        df['DateTime'] = pd.date_range(start='2025-01-01', periods=len(df), freq='1min')
        
        # Take a small sample for debugging (first 10000 rows)
        df_sample = df.head(10000).copy()
        logger.info(f"🎯 Using sample data: {len(df_sample)} rows")
        
        # Debug 1: Check data quality
        logger.info("🔍 DEBUG 1: Data Quality Check")
        logger.info(f"📊 Price range: {df_sample['Close'].min():.2f} - {df_sample['Close'].max():.2f}")
        logger.info(f"📊 Data types: {df_sample[['Open', 'High', 'Low', 'Close']].dtypes.to_dict()}")
        
        # Check for missing values
        missing_values = df_sample[['Open', 'High', 'Low', 'Close']].isnull().sum()
        logger.info(f"📊 Missing values: {missing_values.to_dict()}")
        
        # Debug 2: Calculate technical indicators
        logger.info("🔍 DEBUG 2: Technical Indicators Calculation")
        
        df_sample['SMA_20'] = df_sample['Close'].rolling(window=20).mean()
        df_sample['SMA_50'] = df_sample['Close'].rolling(window=50).mean()
        df_sample['EMA_12'] = df_sample['Close'].ewm(span=12).mean()
        df_sample['EMA_26'] = df_sample['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df_sample['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_sample['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df_sample['MACD'] = df_sample['EMA_12'] - df_sample['EMA_26']
        df_sample['MACD_Signal'] = df_sample['MACD'].ewm(span=9).mean()
        
        # Check indicators
        logger.info(f"📊 SMA_20 range: {df_sample['SMA_20'].min():.2f} - {df_sample['SMA_20'].max():.2f}")
        logger.info(f"📊 RSI range: {df_sample['RSI'].min():.1f} - {df_sample['RSI'].max():.1f}")
        logger.info(f"📊 MACD range: {df_sample['MACD'].min():.4f} - {df_sample['MACD'].max():.4f}")
        
        # Debug 3: Generate trading signals
        logger.info("🔍 DEBUG 3: Trading Signals Generation")
        
        df_sample['Signal'] = 0
        
        # Buy signal conditions
        buy_condition = (
            (df_sample['SMA_20'] > df_sample['SMA_50']) & 
            (df_sample['RSI'] < 70) & 
            (df_sample['MACD'] > df_sample['MACD_Signal'])
        )
        
        # Sell signal conditions
        sell_condition = (
            (df_sample['SMA_20'] < df_sample['SMA_50']) & 
            (df_sample['RSI'] > 30) & 
            (df_sample['MACD'] < df_sample['MACD_Signal'])
        )
        
        df_sample.loc[buy_condition, 'Signal'] = 1
        df_sample.loc[sell_condition, 'Signal'] = -1
        
        # Count signals
        buy_signals = (df_sample['Signal'] == 1).sum()
        sell_signals = (df_sample['Signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        logger.info(f"🔥 Buy signals: {buy_signals}")
        logger.info(f"🔥 Sell signals: {sell_signals}")
        logger.info(f"🔥 Total signals: {total_signals}")
        
        # Debug 4: Analyze signal conditions individually
        logger.info("🔍 DEBUG 4: Individual Signal Conditions Analysis")
        
        # Check each condition separately
        sma_up = (df_sample['SMA_20'] > df_sample['SMA_50']).sum()
        sma_down = (df_sample['SMA_20'] < df_sample['SMA_50']).sum()
        rsi_low = (df_sample['RSI'] < 70).sum()
        rsi_high = (df_sample['RSI'] > 30).sum()
        macd_up = (df_sample['MACD'] > df_sample['MACD_Signal']).sum()
        macd_down = (df_sample['MACD'] < df_sample['MACD_Signal']).sum()
        
        logger.info(f"📊 SMA_20 > SMA_50: {sma_up} rows ({sma_up/len(df_sample)*100:.1f}%)")
        logger.info(f"📊 SMA_20 < SMA_50: {sma_down} rows ({sma_down/len(df_sample)*100:.1f}%)")
        logger.info(f"📊 RSI < 70: {rsi_low} rows ({rsi_low/len(df_sample)*100:.1f}%)")
        logger.info(f"📊 RSI > 30: {rsi_high} rows ({rsi_high/len(df_sample)*100:.1f}%)")
        logger.info(f"📊 MACD > Signal: {macd_up} rows ({macd_up/len(df_sample)*100:.1f}%)")
        logger.info(f"📊 MACD < Signal: {macd_down} rows ({macd_down/len(df_sample)*100:.1f}%)")
        
        # Debug 5: Check signal strength and timing
        if total_signals > 0:
            logger.info("🔍 DEBUG 5: Signal Details Analysis")
            
            # Find first few signals
            signal_rows = df_sample[df_sample['Signal'] != 0].head(10)
            
            for idx, row in signal_rows.iterrows():
                signal_type = "BUY" if row['Signal'] == 1 else "SELL"
                logger.info(f"🎯 Signal {idx}: {signal_type} at price {row['Close']:.2f}")
                logger.info(f"   SMA_20: {row['SMA_20']:.2f}, SMA_50: {row['SMA_50']:.2f}")
                logger.info(f"   RSI: {row['RSI']:.1f}, MACD: {row['MACD']:.4f}, Signal: {row['MACD_Signal']:.4f}")
        
        # Debug 6: Test position sizing
        logger.info("🔍 DEBUG 6: Position Sizing Analysis")
        
        initial_capital = 100.0
        risk_per_trade = 0.02  # 2%
        
        # Test position sizing for different scenarios
        test_prices = [2000.0, 2100.0, 2200.0]
        test_stops = [1998.0, 2098.0, 2198.0]
        
        for price, stop in zip(test_prices, test_stops):
            risk_amount = initial_capital * risk_per_trade
            stop_distance = abs(price - stop)
            lot_size = risk_amount / (stop_distance * 10)
            
            # Apply limits
            min_lot_size = 0.01
            max_lot_size = 10.0
            lot_size = max(min_lot_size, min(lot_size, max_lot_size))
            
            logger.info(f"💰 Price: ${price:.2f}, Stop: ${stop:.2f}")
            logger.info(f"   Risk: ${risk_amount:.2f}, Distance: {stop_distance:.2f}")
            logger.info(f"   Position Size: {lot_size:.2f} lots")
        
        # Debug 7: Check profit calculation logic
        logger.info("🔍 DEBUG 7: Profit Calculation Logic")
        
        # Simulate a simple trade
        entry_price = 2000.0
        exit_price = 2012.0  # +12 points
        volume = 0.1  # 0.1 lots
        
        profit_points = exit_price - entry_price
        profit_loss = profit_points * volume * 10  # $10 per pip per lot
        commission = 0.30 * volume  # $0.30 per lot
        spread_cost = 2.0 * volume  # 2 point spread
        net_profit = profit_loss - commission - spread_cost
        
        logger.info(f"💰 Trade Example:")
        logger.info(f"   Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
        logger.info(f"   Volume: {volume} lots")
        logger.info(f"   Profit Points: {profit_points:.2f}")
        logger.info(f"   Gross Profit: ${profit_loss:.2f}")
        logger.info(f"   Commission: ${commission:.2f}")
        logger.info(f"   Spread Cost: ${spread_cost:.2f}")
        logger.info(f"   Net Profit: ${net_profit:.2f}")
        
        # Debug 8: Check for NaN values in signals
        logger.info("🔍 DEBUG 8: NaN Values Check")
        
        nan_count = df_sample[['Signal', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']].isnull().sum()
        logger.info(f"📊 NaN values in indicators: {nan_count.to_dict()}")
        
        # Debug 9: Save sample data for manual inspection
        logger.info("🔍 DEBUG 9: Saving Sample Data")
        
        output_file = "/content/drive/MyDrive/ProjectP-1/debug_signals_sample.csv"
        sample_output = df_sample[['DateTime', 'Open', 'High', 'Low', 'Close', 
                                   'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Signal']].copy()
        sample_output.to_csv(output_file, index=False)
        
        logger.info(f"💾 Sample data saved to: {output_file}")
        
        # Summary
        logger.info("🔍 DEBUG SUMMARY:")
        logger.info(f"📊 Data rows analyzed: {len(df_sample)}")
        logger.info(f"🎯 Total signals generated: {total_signals}")
        logger.info(f"🔥 Signal rate: {total_signals/len(df_sample)*100:.2f}%")
        
        if total_signals == 0:
            logger.warning("⚠️ NO SIGNALS GENERATED - This explains 0% win rate!")
            logger.warning("⚠️ Possible issues:")
            logger.warning("   1. Signal conditions too strict")
            logger.warning("   2. Market conditions don't match strategy")
            logger.warning("   3. Technical indicators not working as expected")
            logger.warning("   4. Data quality issues")
        
        return total_signals > 0
        
    except Exception as e:
        logger.error(f"❌ Debug analysis error: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔍 Starting Trading Signals Debug Analysis...")
    success = debug_trading_signals()
    
    if success:
        print("✅ Debug analysis completed successfully!")
    else:
        print("❌ Debug analysis failed!")
