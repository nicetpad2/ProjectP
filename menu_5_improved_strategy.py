#!/usr/bin/env python3
"""
🎯 MENU 5 - IMPROVED TRADING STRATEGY
ระบบเทรดที่ปรับปรุงใหม่ด้วย M15 timeframe และกลยุทธ์ที่เหมาะสม
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

# Add project path
project_root = '/content/drive/MyDrive/ProjectP-1'
sys.path.append(project_root)

try:
    from core.unified_enterprise_logger import UnifiedEnterpriseLogger
    logger = UnifiedEnterpriseLogger(
        component_name="MENU5_IMPROVED",
        session_id="improved_strategy"
    )
except:
    print("Using basic logger")
    class BasicLogger:
        def info(self, msg): print(f"ℹ️ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def success(self, msg): print(f"✅ {msg}")
    logger = BasicLogger()

@dataclass
class TradeResult:
    entry_time: str
    exit_time: str
    signal_type: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str

class ImprovedTradingStrategy:
    """Improved trading strategy using M15 timeframe with better indicators"""
    
    def __init__(self):
        self.project_paths = self._get_project_paths()
        
    def _get_project_paths(self):
        """Get project paths"""
        return {
            'data': os.path.join(project_root, 'datacsv'),
            'outputs': os.path.join(project_root, 'outputs', 'backtest_results')
        }
    
    def load_m15_data(self) -> pd.DataFrame:
        """Load M15 (15-minute) data for better signal quality"""
        try:
            data_path = os.path.join(self.project_paths['data'], 'XAUUSD_M15.csv')
            logger.info(f"📊 Loading M15 data from: {data_path}")
            
            df = pd.read_csv(data_path)
            logger.info(f"✅ M15 Data loaded: {len(df)} rows")
            
            # Convert columns to numeric
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create datetime index
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], errors='coerce')
            if df['Datetime'].isna().all():
                # If datetime parsing fails, create sequential datetime
                start_date = datetime(2025, 1, 1)
                df['Datetime'] = [start_date + timedelta(minutes=15*i) for i in range(len(df))]
                logger.info("Using sequential datetime for M15 data")
            
            df.set_index('Datetime', inplace=True)
            df = df.dropna()
            
            logger.info(f"📅 M15 Date range: {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load M15 data: {e}")
            raise
    
    def calculate_improved_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate improved technical indicators for M15 timeframe"""
        logger.info("📈 Calculating improved indicators for M15 timeframe...")
        
        # Moving Averages (longer periods for M15)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()  # ~12.5 hours
        df['SMA_200'] = df['Close'].rolling(window=200).mean()  # ~50 hours
        df['EMA_21'] = df['Close'].ewm(span=21).mean()  # ~5.25 hours
        
        # Trend Strength Indicator (ADX-like)
        def calculate_trend_strength(high, low, close, period=14):
            tr = np.maximum(high - low, 
                           np.maximum(np.abs(high - close.shift(1)), 
                                    np.abs(low - close.shift(1))))
            atr = tr.rolling(window=period).mean()
            
            plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                              np.maximum(high - high.shift(1), 0), 0)
            minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                               np.maximum(low.shift(1) - low, 0), 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx, plus_di, minus_di
        
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_trend_strength(df['High'], df['Low'], df['Close'])
        
        # Volume-based momentum (if volume available, otherwise use price momentum)
        df['Price_Momentum'] = df['Close'].pct_change(periods=5) * 100  # 5-period momentum
        
        # Volatility indicator
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean() * 100
        
        # Support/Resistance levels
        df['Highest_High'] = df['High'].rolling(window=50).max()
        df['Lowest_Low'] = df['Low'].rolling(window=50).min()
        
        return df
    
    def generate_improved_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate improved trading signals with multiple confirmations"""
        logger.info("🎯 Generating improved trading signals...")
        
        # Condition 1: Trend Direction (SMA 50 vs SMA 200)
        trend_up = df['SMA_50'] > df['SMA_200']
        trend_down = df['SMA_50'] < df['SMA_200']
        
        # Condition 2: Price above/below EMA 21
        price_above_ema = df['Close'] > df['EMA_21']
        price_below_ema = df['Close'] < df['EMA_21']
        
        # Condition 3: Strong trend (ADX > 25)
        strong_trend = df['ADX'] > 25
        
        # Condition 4: Momentum confirmation
        positive_momentum = df['Price_Momentum'] > 0.1
        negative_momentum = df['Price_Momentum'] < -0.1
        
        # Condition 5: Not in extreme volatility
        normal_volatility = df['Volatility'] < df['Volatility'].quantile(0.8)
        
        # Condition 6: Directional Movement confirmation
        bullish_dm = df['Plus_DI'] > df['Minus_DI']
        bearish_dm = df['Plus_DI'] < df['Minus_DI']
        
        # Generate BUY signals (require ALL conditions)
        buy_signals = (
            trend_up &          # Long-term trend is up
            price_above_ema &   # Price above short-term average
            strong_trend &      # Strong trend present
            positive_momentum & # Positive momentum
            normal_volatility & # Not extreme volatility
            bullish_dm          # Bullish directional movement
        )
        
        # Generate SELL signals (require ALL conditions)
        sell_signals = (
            trend_down &        # Long-term trend is down
            price_below_ema &   # Price below short-term average
            strong_trend &      # Strong trend present
            negative_momentum & # Negative momentum
            normal_volatility & # Not extreme volatility
            bearish_dm          # Bearish directional movement
        )
        
        # Create signal column
        df['Signal'] = 0
        df.loc[buy_signals, 'Signal'] = 1   # Buy
        df.loc[sell_signals, 'Signal'] = -1  # Sell
        
        # Calculate stop loss and take profit (wider for M15)
        df['Stop_Loss'] = np.where(
            df['Signal'] == 1,
            df['Close'] * 0.995,   # 0.5% stop loss for buy
            df['Close'] * 1.005    # 0.5% stop loss for sell
        )
        
        df['Take_Profit'] = np.where(
            df['Signal'] == 1,
            df['Close'] * 1.015,   # 1.5% take profit for buy (3:1 ratio)
            df['Close'] * 0.985    # 1.5% take profit for sell (3:1 ratio)
        )
        
        # Count signals
        buy_count = (df['Signal'] == 1).sum()
        sell_count = (df['Signal'] == -1).sum()
        total_signals = buy_count + sell_count
        
        logger.info(f"📊 Improved Signal Analysis:")
        logger.info(f"   📈 Buy signals: {buy_count}")
        logger.info(f"   📉 Sell signals: {sell_count}")
        logger.info(f"   🎯 Total signals: {total_signals}")
        logger.info(f"   📊 Signal rate: {(total_signals/len(df)*100):.2f}%")
        
        return df
    
    def simulate_improved_trades(self, df: pd.DataFrame) -> List[TradeResult]:
        """Simulate trades with improved strategy"""
        logger.info("🎮 Simulating trades with improved strategy...")
        
        trades = []
        current_position = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check for new signals
            if row['Signal'] != 0 and current_position is None:
                # Open new position
                current_position = {
                    'type': 'buy' if row['Signal'] == 1 else 'sell',
                    'entry_time': row.name,
                    'entry_price': row['Close'],
                    'stop_loss': row['Stop_Loss'],
                    'take_profit': row['Take_Profit']
                }
                continue
            
            # Check for position exit
            if current_position is not None:
                current_price = row['Close']
                exit_reason = None
                
                if current_position['type'] == 'buy':
                    if current_price <= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price >= current_position['take_profit']:
                        exit_reason = 'take_profit'
                else:  # sell position
                    if current_price >= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price <= current_position['take_profit']:
                        exit_reason = 'take_profit'
                
                # Close position if exit condition met
                if exit_reason:
                    # Calculate P&L
                    if current_position['type'] == 'buy':
                        pnl_pct = (current_price - current_position['entry_price']) / current_position['entry_price'] * 100
                    else:
                        pnl_pct = (current_position['entry_price'] - current_price) / current_position['entry_price'] * 100
                    
                    pnl = 100 * pnl_pct / 100  # Assuming $100 position size
                    
                    trade = TradeResult(
                        entry_time=str(current_position['entry_time']),
                        exit_time=str(row.name),
                        signal_type=current_position['type'],
                        entry_price=current_position['entry_price'],
                        exit_price=current_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason
                    )
                    
                    trades.append(trade)
                    current_position = None
        
        return trades
    
    def analyze_results(self, trades: List[TradeResult]) -> dict:
        """Analyze trading results"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in trades)
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Count exit reasons
        stop_loss_exits = sum(1 for t in trades if t.exit_reason == 'stop_loss')
        take_profit_exits = sum(1 for t in trades if t.exit_reason == 'take_profit')
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'stop_loss_exits': stop_loss_exits,
            'take_profit_exits': take_profit_exits
        }
        
        return results
    
    def run_improved_backtest(self):
        """Run complete improved backtest"""
        logger.info("🚀 Starting Improved Trading Strategy Backtest")
        logger.info("🎯 Using M15 timeframe with multi-condition signals")
        
        try:
            # Load M15 data
            df = self.load_m15_data()
            
            # Calculate improved indicators
            df = self.calculate_improved_indicators(df)
            
            # Generate improved signals
            df = self.generate_improved_signals(df)
            
            # Simulate trades
            trades = self.simulate_improved_trades(df)
            
            # Analyze results
            results = self.analyze_results(trades)
            
            # Display results
            logger.success("✅ Improved Strategy Results:")
            logger.success(f"📊 Total Trades: {results['total_trades']}")
            logger.success(f"✅ Winning Trades: {results['winning_trades']}")
            logger.success(f"❌ Losing Trades: {results['losing_trades']}")
            logger.success(f"🎯 Win Rate: {results['win_rate']:.1f}%")
            logger.success(f"💰 Total P&L: ${results['total_pnl']:.2f}")
            logger.success(f"📈 Average Win: ${results['avg_win']:.2f}")
            logger.success(f"📉 Average Loss: ${results['avg_loss']:.2f}")
            logger.success(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
            
            # Exit reason analysis
            logger.info(f"🛑 Stop Loss Exits: {results['stop_loss_exits']}")
            logger.info(f"🎯 Take Profit Exits: {results['take_profit_exits']}")
            
            # Save results
            os.makedirs(self.project_paths['outputs'], exist_ok=True)
            results_file = os.path.join(
                self.project_paths['outputs'], 
                f'improved_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            # Convert trades to serializable format
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'signal_type': trade.signal_type,
                    'entry_price': float(trade.entry_price),
                    'exit_price': float(trade.exit_price),
                    'pnl': float(trade.pnl),
                    'pnl_pct': float(trade.pnl_pct),
                    'exit_reason': trade.exit_reason
                })
            
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'strategy': 'Improved M15 Multi-Condition Strategy',
                'results': results,
                'trades': trade_data
            }
            
            with open(results_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.success(f"📁 Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in improved backtest: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function"""
    print("🎯 IMPROVED TRADING STRATEGY - M15 TIMEFRAME")
    print("=" * 60)
    
    strategy = ImprovedTradingStrategy()
    results = strategy.run_improved_backtest()
    
    if results:
        print(f"\n🎉 IMPROVED STRATEGY SUMMARY:")
        print(f"📊 Total Trades: {results['total_trades']}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"💰 Total P&L: ${results['total_pnl']:.2f}")
        print(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
        
        if results['win_rate'] > 0:
            print("✅ SUCCESS: Strategy shows profitable potential!")
        else:
            print("❌ FAILED: Strategy still needs improvement")
    else:
        print("❌ FAILED: Could not complete backtest")

if __name__ == "__main__":
    main()
