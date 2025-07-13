#!/usr/bin/env python3
"""
🎯 Menu 5 Enhanced Multi-Timeframe Backtest System
Enhanced NICEGOLD Enterprise ProjectP - Multi-Timeframe Backtest System

Features:
- Multi-Timeframe Support (M1, M5, M15, M30, H1, H4, D1)
- Walk Forward Validation
- Professional Capital Management ($100 starting capital)
- Timeframe-optimized trading parameters
- 2% risk per trade with Kelly Criterion
- Compound growth system
- Advanced data conversion from M1 to any timeframe
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.unified_enterprise_logger import get_unified_logger
    from core.project_paths import ProjectPaths
    from core.multi_timeframe_converter import MultiTimeframeConverter, convert_m1_to_timeframe
    logger = get_unified_logger()
    ENTERPRISE_IMPORTS = True
except ImportError:
    logger = logging.getLogger(__name__)
    ENTERPRISE_IMPORTS = False

class TradeResult:
    """Trading result data structure"""
    def __init__(self, entry_time, exit_time, direction, entry_price, exit_price, 
                 volume, profit_loss, commission, spread_cost, net_profit):
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.volume = volume
        self.profit_loss = profit_loss
        self.commission = commission
        self.spread_cost = spread_cost
        self.net_profit = net_profit
        self.is_profitable = net_profit > 0

class WalkForwardWindow:
    """Walk Forward validation window"""
    def __init__(self, train_start, train_end, test_start, test_end):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.trades = []
        self.initial_capital = 0
        self.final_capital = 0
        
    def add_trade(self, trade):
        self.trades.append(trade)
        
    def get_statistics(self):
        if not self.trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'profit_factor': 0,
                'return': 0,
                'final_capital': self.initial_capital
            }
            
        profitable_trades = [t for t in self.trades if t.is_profitable]
        losing_trades = [t for t in self.trades if not t.is_profitable]
        
        total_profit = sum(t.net_profit for t in profitable_trades)
        total_loss = abs(sum(t.net_profit for t in losing_trades))
        
        return {
            'num_trades': len(self.trades),
            'win_rate': len(profitable_trades) / len(self.trades) * 100,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'return': (self.final_capital - self.initial_capital) / self.initial_capital * 100,
            'final_capital': self.final_capital
        }

class EnhancedMultiTimeframeBacktestEngine:
    """Enhanced Multi-Timeframe Backtest Engine"""
    
    def __init__(self, timeframe='M15', initial_capital=100.0, commission=0.0003, 
                 stop_loss=0.015, take_profit=0.03, window_size=20):
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.commission_per_lot = commission * 100  # Commission per lot
        self.spread_points = 0.0002  # Spread in points
        self.stop_loss_pct = stop_loss
        self.take_profit_pct = take_profit
        self.window_size = window_size
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Initialize tracking variables
        self.peak_balance = initial_capital
        self.max_drawdown = 0
        
        logger.info(f"Enhanced Multi-Timeframe Backtest Engine initialized for {timeframe}")
        
    def calculate_indicators(self, data):
        """Calculate technical indicators optimized for timeframe"""
        df = data.copy()
        
        # Timeframe-specific parameters
        if self.timeframe in ['M1', 'M5']:
            fast_sma = 10
            slow_sma = 20
            rsi_period = 14
        elif self.timeframe in ['M15', 'M30']:
            fast_sma = 8
            slow_sma = 21
            rsi_period = 14
        else:  # H1, H4, D1
            fast_sma = 5
            slow_sma = 13
            rsi_period = 14
            
        # Simple Moving Averages
        df['SMA_Fast'] = df['close'].rolling(window=fast_sma).mean()
        df['SMA_Slow'] = df['close'].rolling(window=slow_sma).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Volatility (ATR-like)
        df['High_Low'] = df['high'] - df['low']
        df['Volatility'] = df['High_Low'].rolling(window=14).mean()
        
        return df
        
    def generate_signals(self, data):
        """Generate trading signals based on multiple indicators"""
        df = data.copy()
        
        # Initialize signals
        df['Signal'] = 0  # 0 = No signal, 1 = Buy, -1 = Sell
        
        # Signal conditions
        buy_condition = (
            (df['SMA_Fast'] > df['SMA_Slow']) &  # Uptrend
            (df['RSI'] < 70) &  # Not overbought
            (df['MACD'] > df['MACD_Signal']) &  # MACD bullish
            (df['close'] > df['SMA_Fast'])  # Price above fast SMA
        )
        
        sell_condition = (
            (df['SMA_Fast'] < df['SMA_Slow']) &  # Downtrend
            (df['RSI'] > 30) &  # Not oversold
            (df['MACD'] < df['MACD_Signal']) &  # MACD bearish
            (df['close'] < df['SMA_Fast'])  # Price below fast SMA
        )
        
        df.loc[buy_condition, 'Signal'] = 1
        df.loc[sell_condition, 'Signal'] = -1
        
        return df
        
    def calculate_position_size(self, current_balance, entry_price, stop_loss_price):
        """Calculate position size using Kelly Criterion and risk management"""
        # Calculate risk per trade (2% of balance)
        risk_amount = current_balance * self.risk_per_trade
        
        # Calculate pip value for XAUUSD
        pip_value = 10  # $10 per pip for 1 lot XAUUSD
        
        # Calculate stop loss distance in pips
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        # Calculate position size
        if stop_loss_distance > 0:
            position_size = risk_amount / (stop_loss_distance * pip_value)
            # Limit position size to reasonable range
            position_size = min(position_size, 0.1)  # Max 0.1 lot
            position_size = max(position_size, 0.01)  # Min 0.01 lot
        else:
            position_size = 0.01  # Default minimum
            
        return round(position_size, 2)
        
    def backtest_window(self, data, window):
        """Backtest a single window"""
        # Filter data for testing period
        test_data = data[window.test_start:window.test_end].copy()
        
        if len(test_data) < self.window_size:
            return window
            
        # Calculate indicators
        test_data = self.calculate_indicators(test_data)
        
        # Generate signals
        test_data = self.generate_signals(test_data)
        
        # Initialize tracking
        current_balance = window.initial_capital
        current_position = None
        current_position_data = None
        
        # Simulate trading
        for i in range(self.window_size, len(test_data)):
            row = test_data.iloc[i]
            
            # Check for exit signal if in position
            if current_position is not None:
                exit_signal = False
                exit_price = row['close']
                
                # Check stop loss and take profit
                if current_position == 'BUY':
                    if row['low'] <= current_position_data['stop_loss']:
                        exit_signal = True
                        exit_price = current_position_data['stop_loss']
                    elif row['high'] >= current_position_data['take_profit']:
                        exit_signal = True
                        exit_price = current_position_data['take_profit']
                else:  # SELL
                    if row['high'] >= current_position_data['stop_loss']:
                        exit_signal = True
                        exit_price = current_position_data['stop_loss']
                    elif row['low'] <= current_position_data['take_profit']:
                        exit_signal = True
                        exit_price = current_position_data['take_profit']
                
                # Exit position
                if exit_signal:
                    # Calculate profit/loss
                    if current_position == 'BUY':
                        profit_points = exit_price - current_position_data['entry_price']
                    else:
                        profit_points = current_position_data['entry_price'] - exit_price
                    
                    # Calculate monetary profit/loss
                    profit_loss = profit_points * current_position_data['volume'] * 10
                    
                    # Calculate costs
                    commission = self.commission_per_lot * current_position_data['volume']
                    spread_cost = self.spread_points * current_position_data['volume']
                    
                    # Net profit
                    net_profit = profit_loss - commission - spread_cost
                    
                    # Update balance
                    current_balance += net_profit
                    self.peak_balance = max(self.peak_balance, current_balance)
                    
                    # Calculate drawdown
                    drawdown_pct = (self.peak_balance - current_balance) / self.peak_balance
                    
                    # Create trade result
                    trade_result = TradeResult(
                        entry_time=current_position_data['entry_time'],
                        exit_time=row.name,
                        direction=current_position,
                        entry_price=current_position_data['entry_price'],
                        exit_price=exit_price,
                        volume=current_position_data['volume'],
                        profit_loss=profit_loss,
                        commission=commission,
                        spread_cost=spread_cost,
                        net_profit=net_profit
                    )
                    
                    window.add_trade(trade_result)
                    
                    # Clear position
                    current_position = None
                    current_position_data = None
                    
            # Check for entry signal if not in position
            elif current_position is None and row['Signal'] != 0:
                entry_price = row['close']
                
                # Calculate stop loss and take profit
                if row['Signal'] == 1:  # BUY
                    stop_loss = entry_price * (1 - self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct)
                    position_type = 'BUY'
                else:  # SELL
                    stop_loss = entry_price * (1 + self.stop_loss_pct)
                    take_profit = entry_price * (1 - self.take_profit_pct)
                    position_type = 'SELL'
                
                # Calculate position size
                volume = self.calculate_position_size(current_balance, entry_price, stop_loss)
                
                # Open position
                current_position = position_type
                current_position_data = {
                    'entry_time': row.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'volume': volume
                }
        
        # Update window final capital
        window.final_capital = current_balance
        
        return window
        
    def create_windows(self, data, train_ratio=0.8, step_size=0.2):
        """Create Walk Forward validation windows"""
        windows = []
        data_length = len(data)
        window_size = int(data_length * train_ratio)
        step = int(data_length * step_size)
        
        current_start = 0
        
        while current_start + window_size < data_length:
            train_end = current_start + window_size
            test_start = train_end
            test_end = min(test_start + step, data_length)
            
            if test_end - test_start > self.window_size:
                window = WalkForwardWindow(
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                )
                windows.append(window)
                
            current_start += step
            
        return windows
        
    def run_walk_forward_validation(self, data):
        """Run Walk Forward validation"""
        logger.info(f"Starting Walk Forward validation for {self.timeframe}")
        
        # Create windows
        windows = self.create_windows(data)
        logger.info(f"Created {len(windows)} validation windows")
        
        if not windows:
            logger.error("No validation windows created")
            return None
            
        # Run backtest on each window
        results = []
        cumulative_capital = self.initial_capital
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Set initial capital for this window
            window.initial_capital = cumulative_capital
            
            # Run backtest
            window = self.backtest_window(data, window)
            
            # Update cumulative capital
            cumulative_capital = window.final_capital
            
            # Get statistics
            stats = window.get_statistics()
            results.append(stats)
            
            logger.info(f"Window {i+1} completed: {stats['num_trades']} trades, "
                       f"{stats['win_rate']:.1f}% win rate, "
                       f"{stats['return']:.2f}% return")
        
        # Calculate overall statistics
        total_trades = sum(r['num_trades'] for r in results)
        
        if total_trades > 0:
            avg_win_rate = sum(r['win_rate'] * r['num_trades'] for r in results) / total_trades
            total_return = (cumulative_capital - self.initial_capital) / self.initial_capital * 100
            
            total_profit = sum(r['total_profit'] for r in results)
            total_loss = sum(r['total_loss'] for r in results)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate max drawdown
            capital_curve = [self.initial_capital]
            for result in results:
                capital_curve.append(result['final_capital'])
            
            peak = capital_curve[0]
            max_drawdown = 0
            for capital in capital_curve:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            avg_win_rate = 0
            total_return = 0
            profit_factor = 0
            max_drawdown = 0
        
        final_results = {
            'timeframe': self.timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': cumulative_capital,
            'total_return': total_return,
            'num_trades': total_trades,
            'win_rate': avg_win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'windows': results
        }
        
        logger.info(f"Walk Forward validation completed: {total_return:.2f}% return")
        return final_results

class Menu5EnhancedMultiTimeframeBacktest:
    """Enhanced Multi-Timeframe Backtest System"""
    
    def __init__(self):
        self.paths = ProjectPaths()
        self.logger = logger
        self.available_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        self.timeframe_settings = {
            'M1': {'name': '1 Minute', 'window_size': 100, 'stop_loss': 0.005, 'take_profit': 0.01},
            'M5': {'name': '5 Minutes', 'window_size': 50, 'stop_loss': 0.01, 'take_profit': 0.02},
            'M15': {'name': '15 Minutes', 'window_size': 20, 'stop_loss': 0.015, 'take_profit': 0.03},
            'M30': {'name': '30 Minutes', 'window_size': 15, 'stop_loss': 0.02, 'take_profit': 0.04},
            'H1': {'name': '1 Hour', 'window_size': 10, 'stop_loss': 0.025, 'take_profit': 0.05},
            'H4': {'name': '4 Hours', 'window_size': 8, 'stop_loss': 0.03, 'take_profit': 0.06},
            'D1': {'name': '1 Day', 'window_size': 5, 'stop_loss': 0.04, 'take_profit': 0.08}
        }
        
        print("\n🎯 MENU 5 ENHANCED MULTI-TIMEFRAME BACKTEST SYSTEM")
        print("=" * 70)
        print("📊 Enhanced System: Walk Forward Validation")
        print("💰 Starting Capital: $100")
        print("📈 Multi-Timeframe Support:")
        for tf, settings in self.timeframe_settings.items():
            print(f"   📍 {tf}: {settings['name']}")
        print("🎯 Timeframe-Optimized Parameters")
        print("📊 Advanced Signal Generation")
        print("=" * 70)
        
    def load_and_convert_data(self, timeframe='M15'):
        """Load and convert data to specified timeframe"""
        try:
            # Load M1 data
            csv_path = self.paths.datacsv / "XAUUSD_M1.csv"
            self.logger.info(f"Loading M1 data from: {csv_path}")
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")
                
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} rows of M1 data")
            
            # Prepare data for conversion
            # Create DateTime column if not exists
            if 'DateTime' not in df.columns:
                if 'Date' in df.columns and 'Timestamp' in df.columns:
                    # Fix Date format (25630501 -> 2020-05-01)
                    # Convert Thai year format to standard format
                    df['Date'] = df['Date'].astype(str)
                    
                    # Extract year, month, day
                    year = df['Date'].str[:4].astype(int) - 543  # Convert Thai year to Western year
                    month = df['Date'].str[4:6] 
                    day = df['Date'].str[6:8]
                    
                    # Create date string
                    date_str = year.astype(str) + '-' + month + '-' + day
                    
                    # Create DateTime column from Date and Timestamp
                    df['DateTime'] = pd.to_datetime(date_str + ' ' + df['Timestamp'].astype(str), errors='coerce')
                    
                    # Remove any rows with NaT (Not a Time) values
                    df = df.dropna(subset=['DateTime'])
                    
                elif 'Date' in df.columns:
                    # Handle Date only format
                    df['Date'] = df['Date'].astype(str)
                    
                    # Extract year, month, day
                    year = df['Date'].str[:4].astype(int) - 543  # Convert Thai year to Western year
                    month = df['Date'].str[4:6]
                    day = df['Date'].str[6:8]
                    
                    # Create date string
                    date_str = year.astype(str) + '-' + month + '-' + day
                    df['DateTime'] = pd.to_datetime(date_str, errors='coerce')
                    df = df.dropna(subset=['DateTime'])
                else:
                    # Use index as datetime if no date columns
                    df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
            
            # Standardize column names to lowercase
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            # Convert to requested timeframe if not M1
            if timeframe != 'M1':
                if ENTERPRISE_IMPORTS:
                    self.logger.info(f"Converting M1 to {timeframe} timeframe...")
                    result = convert_m1_to_timeframe(df, timeframe)
                    df = result.data  # Extract DataFrame from ConversionResult
                    self.logger.info(f"Converted to {len(df)} rows of {timeframe} data")
                else:
                    self.logger.warning(f"Multi-timeframe converter not available, using M1 data")
            
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in columns: {list(df.columns)}")
            
            # Ensure numeric data
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN values
            df = df.dropna(subset=required_columns)
            
            # Set DateTime as index for easier processing
            if 'DateTime' in df.columns:
                df.set_index('DateTime', inplace=True)
            
            self.logger.info(f"Data validation complete: {len(df)} rows ready for backtesting")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def run_backtest(self, timeframe='M15', initial_capital=100.0):
        """Run backtest with specified timeframe and capital"""
        try:
            # Load and convert data
            data = self.load_and_convert_data(timeframe)
            
            # Get timeframe settings
            settings = self.timeframe_settings[timeframe]
            
            print(f"\n📊 Running enhanced backtest on {settings['name']} timeframe")
            print(f"💰 Starting Capital: ${initial_capital:.2f}")
            print(f"📈 Data Points: {len(data)}")
            print(f"🎯 Window Size: {settings['window_size']}")
            print(f"🛑 Stop Loss: {settings['stop_loss']*100:.1f}%")
            print(f"🎯 Take Profit: {settings['take_profit']*100:.1f}%")
            
            # Create enhanced backtest engine
            engine = EnhancedMultiTimeframeBacktestEngine(
                timeframe=timeframe,
                initial_capital=initial_capital,
                commission=0.0003,
                stop_loss=settings['stop_loss'],
                take_profit=settings['take_profit'],
                window_size=settings['window_size']
            )
            
            # Run backtest
            self.logger.info(f"Starting Enhanced Walk Forward Validation on {timeframe} timeframe")
            results = engine.run_walk_forward_validation(data)
            
            # Display results
            self.display_results(results, timeframe)
            
            # Save results
            self.save_results(results, timeframe)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            print(f"❌ Backtest failed: {str(e)}")
            return None
            
    def display_results(self, results, timeframe):
        """Display backtest results in beautiful format"""
        if not results or not results.get('windows'):
            print("\n❌ No results to display")
            return
            
        print(f"\n🎉 ENHANCED BACKTEST RESULTS - {timeframe} TIMEFRAME")
        print("=" * 70)
        
        # Overall statistics
        final_capital = results['final_capital']
        total_return = results['total_return']
        num_trades = results['num_trades']
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        max_drawdown = results['max_drawdown']
        
        print(f"💰 Final Capital: ${final_capital:.2f}")
        print(f"📈 Total Return: {total_return:.2f}%")
        print(f"🎯 Total Trades: {num_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}%")
        print(f"📊 Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        
        # Performance rating
        if total_return > 20:
            rating = "🌟 EXCELLENT"
        elif total_return > 10:
            rating = "✅ GOOD"
        elif total_return > 0:
            rating = "📈 POSITIVE"
        else:
            rating = "📉 NEGATIVE"
            
        print(f"🎯 Performance Rating: {rating}")
        
        # Window-by-window results
        print(f"\n📊 ENHANCED WALK FORWARD VALIDATION RESULTS")
        print("=" * 70)
        
        for i, window in enumerate(results['windows'], 1):
            print(f"🎯 Window {i}: Capital: ${window['final_capital']:.2f}, "
                  f"Return: {window['return']:.2f}%, "
                  f"Trades: {window['num_trades']}, "
                  f"Win Rate: {window['win_rate']:.1f}%")
        
        # Best and worst windows
        if len(results['windows']) > 1:
            best_window = max(results['windows'], key=lambda x: x['return'])
            worst_window = min(results['windows'], key=lambda x: x['return'])
            
            print(f"\n🏆 Best Window: {best_window['return']:.2f}% return")
            print(f"😢 Worst Window: {worst_window['return']:.2f}% return")
            
    def save_results(self, results, timeframe):
        """Save backtest results to file"""
        try:
            # Create output directory
            output_dir = self.paths.outputs / "enhanced_backtest_results"
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_backtest_{timeframe}_{timestamp}.json"
            filepath = output_dir / filename
            
            # Add metadata
            results_with_meta = {
                'metadata': {
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'system': 'Enhanced Multi-Timeframe Backtest with Walk Forward Validation',
                    'version': '2.0'
                },
                'results': results
            }
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(results_with_meta, f, indent=2)
                
            self.logger.info(f"Results saved to: {filepath}")
            print(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            print(f"❌ Error saving results: {str(e)}")
            
    def run_all_timeframes(self):
        """Run backtest on all available timeframes"""
        print("\n🚀 RUNNING ENHANCED BACKTEST ON ALL TIMEFRAMES")
        print("=" * 70)
        
        all_results = {}
        
        for timeframe in self.available_timeframes:
            print(f"\n📊 Testing {timeframe} timeframe...")
            try:
                results = self.run_backtest(timeframe)
                if results:
                    all_results[timeframe] = results
                    print(f"✅ {timeframe} completed successfully")
                else:
                    print(f"❌ {timeframe} failed")
                    
            except Exception as e:
                print(f"❌ {timeframe} failed: {str(e)}")
                self.logger.error(f"Timeframe {timeframe} failed: {str(e)}")
                
        # Compare results
        self.compare_timeframes(all_results)
        
        return all_results
        
    def compare_timeframes(self, all_results):
        """Compare results across all timeframes"""
        if not all_results:
            print("\n❌ No results to compare")
            return
            
        print(f"\n📊 ENHANCED TIMEFRAME COMPARISON")
        print("=" * 90)
        print(f"{'Timeframe':<10} {'Final Capital':<15} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15}")
        print("-" * 90)
        
        for timeframe, results in all_results.items():
            final_capital = results['final_capital']
            total_return = results['total_return']
            num_trades = results['num_trades']
            win_rate = results['win_rate']
            profit_factor = results['profit_factor']
            
            print(f"{timeframe:<10} ${final_capital:<14.2f} {total_return:<9.2f}% {num_trades:<7} {win_rate:<9.1f}% {profit_factor:<14.2f}")
            
        # Find best timeframe
        best_timeframe = max(all_results.keys(), key=lambda x: all_results[x]['total_return'])
        best_return = all_results[best_timeframe]['total_return']
        
        print(f"\n🏆 Best Timeframe: {best_timeframe} with {best_return:.2f}% return")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_return > 10:
            print(f"✅ {best_timeframe} timeframe shows excellent performance")
        elif best_return > 0:
            print(f"📈 {best_timeframe} timeframe shows positive performance")
        else:
            print("⚠️ Consider strategy optimization or different market conditions")
            
    def run_menu(self):
        """Run the Menu 5 Enhanced Multi-Timeframe Backtest system"""
        try:
            print("\n🎯 MENU 5 ENHANCED MULTI-TIMEFRAME BACKTEST SYSTEM")
            print("=" * 70)
            print("1. 📊 M15 Enhanced Backtest (Recommended)")
            print("2. 🚀 Run All Timeframes")
            print("3. 🎯 Custom Timeframe")
            print("4. 📈 M1 vs M15 vs H1 Comparison")
            print("5. 🔧 Timeframe Performance Analysis")
            print("6. 🔙 Back to Main Menu")
            print("=" * 70)
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                print("\n🎯 Running Enhanced M15 Backtest (Recommended for Gold Trading)")
                self.run_backtest('M15')
                
            elif choice == '2':
                print("\n🚀 Running Enhanced Backtest on All Timeframes")
                self.run_all_timeframes()
                
            elif choice == '3':
                print("\n🎯 Available Timeframes:")
                for i, tf in enumerate(self.available_timeframes, 1):
                    settings = self.timeframe_settings[tf]
                    print(f"{i}. {tf} - {settings['name']}")
                    
                try:
                    tf_choice = int(input("\nSelect timeframe (1-7): ").strip())
                    if 1 <= tf_choice <= len(self.available_timeframes):
                        timeframe = self.available_timeframes[tf_choice - 1]
                        self.run_backtest(timeframe)
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid input")
                    
            elif choice == '4':
                print("\n📈 Multi-Timeframe Comparison")
                results = {}
                for tf in ['M1', 'M15', 'H1']:
                    print(f"\n📊 Testing {tf}...")
                    result = self.run_backtest(tf)
                    if result:
                        results[tf] = result
                        
                if results:
                    self.compare_timeframes(results)
                    
            elif choice == '5':
                print("\n🔧 Timeframe Performance Analysis")
                print("Running comprehensive analysis on selected timeframes...")
                
                # Test key timeframes
                key_timeframes = ['M15', 'H1', 'H4']
                results = {}
                
                for tf in key_timeframes:
                    print(f"\n📊 Analyzing {tf}...")
                    result = self.run_backtest(tf)
                    if result:
                        results[tf] = result
                        
                if results:
                    self.compare_timeframes(results)
                    
            elif choice == '6':
                print("\n🔙 Returning to Main Menu")
                return
                
            else:
                print("❌ Invalid choice")
                
        except Exception as e:
            self.logger.error(f"Menu execution error: {str(e)}")
            print(f"❌ Menu error: {str(e)}")
            
        # Ask if user wants to continue
        try:
            continue_choice = input("\nContinue with Menu 5? (y/n): ").strip().lower()
            if continue_choice == 'y':
                self.run_menu()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

def main():
    """Main function for testing"""
    menu = Menu5EnhancedMultiTimeframeBacktest()
    menu.run_menu()

if __name__ == "__main__":
    main()
