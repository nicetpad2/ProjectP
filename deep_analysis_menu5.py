#!/usr/bin/env python3
"""
🔍 DEEP ANALYSIS MENU 5 - COMPREHENSIVE DEBUGGING
การวิเคราะห์เชิงลึกเพื่อหาสาเหตุจริงที่ Menu 5 ไม่มีการเทรดที่ได้กำไร
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

# Add project path
project_root = '/content/drive/MyDrive/ProjectP-1'
sys.path.append(project_root)

try:
    from core.unified_enterprise_logger import UnifiedEnterpriseLogger
    logger = UnifiedEnterpriseLogger(
        component_name="DEEP_ANALYSIS_MENU5",
        session_id="deep_analysis"
    )
except:
    print("Using basic logger")
    class BasicLogger:
        def info(self, msg): print(f"ℹ️ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def success(self, msg): print(f"✅ {msg}")
    logger = BasicLogger()

def deep_analysis_trading_system():
    """การวิเคราะห์เชิงลึกระบบเทรด"""
    
    logger.info("🔍 Starting Deep Analysis of Menu 5 Trading System")
    
    # Load data
    try:
        data_path = os.path.join(project_root, 'datacsv', 'XAUUSD_M1.csv')
        logger.info(f"📊 Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"✅ Data loaded: {len(df)} rows")
        
        # Use first 50,000 rows for detailed analysis
        df = df.head(50000).copy()
        logger.info(f"🎯 Using first 50,000 rows for analysis")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return
    
    # Convert columns to numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate technical indicators
    logger.info("📈 Calculating technical indicators...")
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # Remove NaN rows
    df = df.dropna()
    logger.info(f"📊 After removing NaN: {len(df)} rows")
    
    # Generate trading signals (same logic as Menu 5)
    logger.info("🎯 Generating trading signals...")
    
    buy_signals = (
        (df['SMA_20'] > df['SMA_50']) &  # SMA crossover
        (df['RSI'] < 70) &  # Not overbought
        (df['MACD'] > df['MACD_signal'])  # MACD bullish
    )
    
    sell_signals = (
        (df['SMA_20'] < df['SMA_50']) &  # SMA crossover
        (df['RSI'] > 30) &  # Not oversold
        (df['MACD'] < df['MACD_signal'])  # MACD bearish
    )
    
    # Count signals
    buy_count = buy_signals.sum()
    sell_count = sell_signals.sum()
    total_signals = buy_count + sell_count
    
    logger.info(f"📊 Signal Analysis:")
    logger.info(f"   📈 Buy signals: {buy_count}")
    logger.info(f"   📉 Sell signals: {sell_count}")
    logger.info(f"   🎯 Total signals: {total_signals}")
    logger.info(f"   📊 Signal rate: {(total_signals/len(df)*100):.2f}%")
    
    # Analyze market volatility
    logger.info("📊 Analyzing market volatility...")
    
    # Calculate price movements
    df['price_change'] = df['Close'].pct_change() * 100
    df['high_low_range'] = ((df['High'] - df['Low']) / df['Close'] * 100)
    
    avg_volatility = df['high_low_range'].mean()
    max_volatility = df['high_low_range'].max()
    min_volatility = df['high_low_range'].min()
    
    logger.info(f"💫 Volatility Analysis:")
    logger.info(f"   📊 Average daily range: {avg_volatility:.4f}%")
    logger.info(f"   📈 Maximum daily range: {max_volatility:.4f}%")
    logger.info(f"   📉 Minimum daily range: {min_volatility:.4f}%")
    
    # Test different stop loss and take profit levels
    logger.info("🧪 Testing different Stop Loss/Take Profit levels...")
    
    stop_loss_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    take_profit_ratios = [2.0, 2.5, 3.0]  # Risk:Reward ratios
    
    results = []
    
    for sl in stop_loss_levels:
        for tp_ratio in take_profit_ratios:
            tp = sl * tp_ratio
            
            # Simulate trades
            wins = 0
            losses = 0
            total_pnl = 0
            
            signal_indices = df[buy_signals | sell_signals].index
            
            for idx in signal_indices:
                if idx >= len(df) - 100:  # Skip near end
                    continue
                    
                is_buy = buy_signals.iloc[idx] if idx < len(buy_signals) else False
                entry_price = df.iloc[idx]['Close']
                
                # Check next 100 bars for exit
                for i in range(1, 101):
                    if idx + i >= len(df):
                        break
                        
                    current_price = df.iloc[idx + i]['Close']
                    
                    if is_buy:
                        # Buy trade
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        
                        if pnl_pct <= -sl:  # Stop loss hit
                            losses += 1
                            total_pnl -= sl
                            break
                        elif pnl_pct >= tp:  # Take profit hit
                            wins += 1
                            total_pnl += tp
                            break
                    else:
                        # Sell trade
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        if pnl_pct <= -sl:  # Stop loss hit
                            losses += 1
                            total_pnl -= sl
                            break
                        elif pnl_pct >= tp:  # Take profit hit
                            wins += 1
                            total_pnl += tp
                            break
            
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            results.append({
                'stop_loss': sl,
                'take_profit': tp,
                'risk_reward': tp_ratio,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            })
    
    # Display results
    logger.info("📊 Stop Loss / Take Profit Analysis Results:")
    logger.info("=" * 80)
    logger.info(f"{'SL%':<5} {'TP%':<5} {'R:R':<5} {'Trades':<7} {'Wins':<5} {'Losses':<7} {'Win%':<6} {'PnL%':<8}")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(f"{result['stop_loss']:<5.1f} {result['take_profit']:<5.1f} {result['risk_reward']:<5.1f} "
                   f"{result['total_trades']:<7} {result['wins']:<5} {result['losses']:<7} "
                   f"{result['win_rate']:<6.1f} {result['total_pnl']:<8.2f}")
    
    # Find best performing parameters
    profitable_results = [r for r in results if r['total_pnl'] > 0]
    
    if profitable_results:
        best_result = max(profitable_results, key=lambda x: x['total_pnl'])
        logger.success("🎯 Best performing parameters found:")
        logger.success(f"   Stop Loss: {best_result['stop_loss']}%")
        logger.success(f"   Take Profit: {best_result['take_profit']}%")
        logger.success(f"   Risk:Reward: 1:{best_result['risk_reward']}")
        logger.success(f"   Win Rate: {best_result['win_rate']:.1f}%")
        logger.success(f"   Total PnL: {best_result['total_pnl']:.2f}%")
    else:
        logger.warning("⚠️ No profitable parameter combinations found!")
        logger.warning("💡 This suggests the trading strategy itself may need revision")
    
    # Analyze signal timing
    logger.info("🕐 Analyzing signal timing and market conditions...")
    
    # Check if signals occur during high volatility periods
    signal_df = df[buy_signals | sell_signals].copy()
    if len(signal_df) > 0:
        avg_signal_volatility = signal_df['high_low_range'].mean()
        logger.info(f"📊 Average volatility during signals: {avg_signal_volatility:.4f}%")
        logger.info(f"📊 Overall market volatility: {avg_volatility:.4f}%")
        
        if avg_signal_volatility > avg_volatility * 1.5:
            logger.warning("⚠️ Signals occur during high volatility periods!")
            logger.warning("💡 Consider filtering signals during extreme volatility")
    
    # Check indicator values at signal points
    if len(signal_df) > 0:
        logger.info("📈 Technical indicator analysis at signal points:")
        logger.info(f"   RSI range: {signal_df['RSI'].min():.1f} - {signal_df['RSI'].max():.1f}")
        logger.info(f"   MACD range: {signal_df['MACD'].min():.4f} - {signal_df['MACD'].max():.4f}")
    
    # Save detailed analysis
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'data_points_analyzed': len(df),
        'total_signals': total_signals,
        'buy_signals': int(buy_count),
        'sell_signals': int(sell_count),
        'signal_rate_percent': float(total_signals/len(df)*100),
        'volatility_analysis': {
            'average_daily_range_percent': float(avg_volatility),
            'max_daily_range_percent': float(max_volatility),
            'min_daily_range_percent': float(min_volatility)
        },
        'stop_loss_take_profit_results': results,
        'best_parameters': best_result if profitable_results else None,
        'recommendations': []
    }
    
    # Generate recommendations
    if not profitable_results:
        analysis_results['recommendations'].extend([
            "Strategy fundamentals need revision - no profitable parameter combinations found",
            "Consider using different technical indicators",
            "Implement trend filtering mechanisms",
            "Add volatility-based position sizing",
            "Consider using machine learning for signal generation"
        ])
    else:
        analysis_results['recommendations'].extend([
            f"Use Stop Loss: {best_result['stop_loss']}% and Take Profit: {best_result['take_profit']}%",
            "Implement the identified profitable parameters",
            "Consider adding volatility filters",
            "Monitor performance in different market conditions"
        ])
    
    # Save results
    output_path = os.path.join(project_root, 'outputs', 'analysis')
    os.makedirs(output_path, exist_ok=True)
    
    results_file = os.path.join(output_path, f'deep_analysis_menu5_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.success(f"✅ Deep analysis completed!")
    logger.success(f"📁 Results saved to: {results_file}")
    
    return analysis_results

if __name__ == "__main__":
    try:
        results = deep_analysis_trading_system()
        
        print("\n" + "="*60)
        print("🎯 DEEP ANALYSIS SUMMARY")
        print("="*60)
        
        if results and results.get('best_parameters'):
            best = results['best_parameters']
            print(f"✅ PROFITABLE PARAMETERS FOUND:")
            print(f"   🎯 Stop Loss: {best['stop_loss']}%")
            print(f"   🎯 Take Profit: {best['take_profit']}%")
            print(f"   📊 Win Rate: {best['win_rate']:.1f}%")
            print(f"   💰 Total PnL: {best['total_pnl']:.2f}%")
        else:
            print("❌ NO PROFITABLE PARAMETERS FOUND")
            print("💡 The trading strategy needs fundamental revision")
        
        print("\n📋 RECOMMENDATIONS:")
        if results and results.get('recommendations'):
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n🔍 CONCLUSION:")
        if results and results.get('best_parameters'):
            print("✅ Problem can be solved with parameter optimization")
        else:
            print("❌ Fundamental strategy issues - deeper revision needed")
            
    except Exception as e:
        print(f"❌ Error in deep analysis: {e}")
        import traceback
        traceback.print_exc()
