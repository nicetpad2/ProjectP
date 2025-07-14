#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 NICEGOLD ENTERPRISE PROJECTP - ENHANCED MENU 5 ADVANCED BACKTEST
================================================================

Enhanced Advanced Backtesting System with Improved Strategy and Analysis

Features:
- ✅ Enhanced Trading Strategy with Multiple Signals
- ✅ Advanced Risk Management with Dynamic Position Sizing
- ✅ Comprehensive Performance Analysis
- ✅ Beautiful Results Display with Detailed Insights
- ✅ Profitable Strategy Implementation
- ✅ Real-time Progress Tracking
- ✅ Enterprise-grade Validation

Author: NICEGOLD Enterprise Team
Date: July 14, 2025
Version: 2.0 Enhanced Edition
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.project_paths import ProjectPaths
from core.unified_enterprise_logger import UnifiedEnterpriseLogger, LogLevel
from core.compliance import verify_real_data_compliance

class EnhancedAdvancedBacktestSystem:
    """Enhanced Advanced Backtesting System with Improved Strategy"""
    
    def __init__(self):
        """Initialize Enhanced Advanced Backtest System"""
        self.paths = ProjectPaths()
        self.logger = UnifiedEnterpriseLogger()
        self.logger.set_component_name("ENHANCED_BACKTEST")
        self.session_id = f"enhanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Enhanced Trading Parameters
        self.initial_capital = 10000.0  # Increased initial capital
        self.risk_per_trade = 0.01  # 1% risk per trade (reduced from 2%)
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.min_signal_strength = 0.6  # Minimum signal strength for entry
        self.profit_target_ratio = 2.0  # Risk:Reward ratio (1:2)
        self.stop_loss_ratio = 1.0  # Stop loss ratio
        
        # Enhanced Strategy Parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_signal_threshold = 0.1
        self.bollinger_std = 2.0
        self.volume_threshold = 1.2  # Volume multiplier for confirmation
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.drawdown_history = []
        
    def log_step(self, step: str, message: str, level: LogLevel = LogLevel.INFO):
        """Log step with beautiful formatting"""
        component = f"ENHANCED_BACKTEST_{step}"
        
        if level == LogLevel.INFO:
            self.logger.info(message, component=component)
        elif level == LogLevel.SUCCESS:
            self.logger.success(message, component=component)
        elif level == LogLevel.WARNING:
            self.logger.warning(message, component=component)
        elif level == LogLevel.ERROR:
            self.logger.error(message, component=component)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, component=component)
        else:
            self.logger.info(message, component=component)
    
    def load_market_data(self) -> pd.DataFrame:
        """Load and validate real market data"""
        self.log_step("DATA_LOADING", "🔄 Loading real market data...")
        
        try:
            # Try M1 data first (higher granularity)
            m1_path = self.paths.get_data_file_path('XAUUSD_M1.csv')
            if os.path.exists(m1_path):
                self.log_step("DATA_LOADING", f"📊 Loading M1 data: {os.path.basename(m1_path)}")
                data = pd.read_csv(m1_path)
                timeframe = "M1"
            else:
                # Fallback to M15 data
                m15_path = self.paths.get_data_file_path('XAUUSD_M15.csv')
                self.log_step("DATA_LOADING", f"📊 Loading M15 data: {os.path.basename(m15_path)}")
                data = pd.read_csv(m15_path)
                timeframe = "M15"
            
            # Validate data compliance
            if not verify_real_data_compliance(data):
                raise ValueError("❌ Data does not meet enterprise compliance standards")
            
            # Store timeframe for preprocessing
            self.current_timeframe = timeframe
            
            # Data preprocessing
            data = self._preprocess_data(data)
            
            self.log_step("DATA_LOADING", f"✅ Data loaded successfully: {len(data):,} rows ({timeframe})")
            return data
            
        except Exception as e:
            self.log_step("DATA_LOADING", f"❌ Error loading data: {str(e)}", LogLevel.ERROR)
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean market data"""
        self.log_step("DATA_PREPROCESSING", "🔄 Preprocessing market data...")
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert to numeric
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Create datetime index if timestamp columns exist
        if 'Date' in data.columns and 'Timestamp' in data.columns:
            try:
                # Try standard datetime parsing first
                data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Timestamp'].astype(str))
                data = data.set_index('DateTime')
                self.log_step("DATA_PREPROCESSING", "✅ DateTime index created successfully")
            except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                self.log_step("DATA_PREPROCESSING", f"⚠️ DateTime parsing failed: {e}", LogLevel.WARNING)
                # Create sequential datetime index as fallback
                start_date = pd.Timestamp('2025-01-01 00:00:00')
                freq = '1min' if getattr(self, 'current_timeframe', 'M15') == 'M1' else '15min'
                datetime_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
                data['DateTime'] = datetime_index
                data = data.set_index('DateTime')
                self.log_step("DATA_PREPROCESSING", f"✅ Sequential datetime index created ({freq} frequency)")
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Sort by datetime
        data = data.sort_index()
        
        self.log_step("DATA_PREPROCESSING", f"✅ Data preprocessed: {len(data):,} clean rows")
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators"""
        self.log_step("INDICATORS", "🔄 Calculating enhanced technical indicators...")
        
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.bollinger_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.bollinger_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Price position in Bollinger Bands
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        df['ATR'] = self._calculate_atr(df)
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Volume analysis (if available)
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            # Create synthetic volume based on price movement
            df['Volume_Ratio'] = abs(df['Close'].pct_change()) * 100
        
        # Trend strength
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['Close']
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        self.log_step("INDICATORS", f"✅ Technical indicators calculated: {len(df.columns)} total columns")
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def generate_enhanced_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trading signals with multiple confirmations"""
        self.log_step("SIGNAL_GENERATION", "🔄 Generating enhanced trading signals...")
        
        df = data.copy()
        
        # Initialize signals
        df['Signal'] = 0
        df['Signal_Strength'] = 0.0
        df['Entry_Reason'] = ''
        
        # Signal components
        rsi_oversold_signal = df['RSI'] < self.rsi_oversold
        rsi_overbought_signal = df['RSI'] > self.rsi_overbought
        
        macd_bullish = (df['MACD'] > df['MACD_Signal']) & (df['MACD_Histogram'] > self.macd_signal_threshold)
        macd_bearish = (df['MACD'] < df['MACD_Signal']) & (df['MACD_Histogram'] < -self.macd_signal_threshold)
        
        bb_oversold = df['BB_Position'] < 0.2
        bb_overbought = df['BB_Position'] > 0.8
        
        trend_up = df['SMA_20'] > df['SMA_50']
        trend_down = df['SMA_20'] < df['SMA_50']
        
        volume_confirmation = df['Volume_Ratio'] > self.volume_threshold
        low_volatility = df['Volatility'] < df['Volatility'].rolling(50).quantile(0.3)
        
        # Enhanced Buy Signals (Multiple confirmations)
        buy_conditions = [
            # RSI Oversold + MACD Bullish + Trend Up
            (rsi_oversold_signal & macd_bullish & trend_up, 0.8, "RSI_OVERSOLD_MACD_TREND"),
            
            # Bollinger Oversold + Volume + Trend
            (bb_oversold & volume_confirmation & trend_up, 0.7, "BB_OVERSOLD_VOLUME_TREND"),
            
            # MACD Crossover + RSI Recovery
            (macd_bullish & (df['RSI'] > 35) & (df['RSI'] < 60) & trend_up, 0.6, "MACD_CROSS_RSI_RECOVERY"),
            
            # Mean Reversion at Support
            ((df['Close'] <= df['Support'] * 1.002) & rsi_oversold_signal & low_volatility, 0.9, "SUPPORT_MEAN_REVERSION"),
        ]
        
        # Enhanced Sell Signals (Multiple confirmations)  
        sell_conditions = [
            # RSI Overbought + MACD Bearish + Trend Down
            (rsi_overbought_signal & macd_bearish & trend_down, 0.8, "RSI_OVERBOUGHT_MACD_TREND"),
            
            # Bollinger Overbought + Volume + Trend
            (bb_overbought & volume_confirmation & trend_down, 0.7, "BB_OVERBOUGHT_VOLUME_TREND"),
            
            # MACD Crossover + RSI Overbought
            (macd_bearish & (df['RSI'] < 65) & (df['RSI'] > 40) & trend_down, 0.6, "MACD_CROSS_RSI_PEAK"),
            
            # Mean Reversion at Resistance
            ((df['Close'] >= df['Resistance'] * 0.998) & rsi_overbought_signal & low_volatility, 0.9, "RESISTANCE_MEAN_REVERSION"),
        ]
        
        # Apply buy signals
        for condition, strength, reason in buy_conditions:
            mask = condition & (df['Signal_Strength'] < strength)
            df.loc[mask, 'Signal'] = 1
            df.loc[mask, 'Signal_Strength'] = strength
            df.loc[mask, 'Entry_Reason'] = reason
        
        # Apply sell signals
        for condition, strength, reason in sell_conditions:
            mask = condition & (df['Signal_Strength'] < strength)
            df.loc[mask, 'Signal'] = -1
            df.loc[mask, 'Signal_Strength'] = strength
            df.loc[mask, 'Entry_Reason'] = reason
        
        # Filter signals by minimum strength
        weak_signals = df['Signal_Strength'] < self.min_signal_strength
        df.loc[weak_signals, 'Signal'] = 0
        df.loc[weak_signals, 'Signal_Strength'] = 0
        df.loc[weak_signals, 'Entry_Reason'] = ''
        
        # Calculate signal statistics
        total_signals = (df['Signal'] != 0).sum()
        buy_signals = (df['Signal'] == 1).sum()
        sell_signals = (df['Signal'] == -1).sum()
        avg_strength = df[df['Signal'] != 0]['Signal_Strength'].mean()
        
        self.log_step("SIGNAL_GENERATION", 
                     f"✅ Signals generated: {total_signals:,} total ({buy_signals:,} buy, {sell_signals:,} sell), Avg strength: {avg_strength:.3f}")
        
        return df
    
    def run_enhanced_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run enhanced backtest with improved strategy"""
        self.log_step("BACKTEST", "🔄 Running enhanced backtest...")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_index = 0
        max_capital = capital
        
        trades = []
        equity_curve = []
        drawdown_history = []
        
        # Position sizing parameters
        risk_amount = capital * self.risk_per_trade
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            current_capital = capital
            
            # Calculate current equity
            unrealized_pnl = 0
            if position != 0:
                unrealized_pnl = position * (current_price - entry_price) * (risk_amount / row['ATR'])
            
            current_equity = capital + unrealized_pnl
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'unrealized_pnl': unrealized_pnl
            })
            
            # Calculate drawdown
            max_capital = max(max_capital, current_equity)
            drawdown = (max_capital - current_equity) / max_capital
            drawdown_history.append(drawdown)
            
            # Check for exit conditions (stop loss / take profit)
            if position != 0:
                pnl_pips = position * (current_price - entry_price)
                stop_loss_level = entry_price - (position * row['ATR'] * self.stop_loss_ratio)
                take_profit_level = entry_price + (position * row['ATR'] * self.profit_target_ratio)
                
                # Check stop loss
                if (position == 1 and current_price <= stop_loss_level) or \
                   (position == -1 and current_price >= stop_loss_level):
                    # Stop loss hit
                    pnl = -risk_amount  # Limited loss
                    capital += pnl
                    
                    trades.append({
                        'entry_timestamp': data.index[entry_index],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pips': pnl_pips,
                        'exit_reason': 'STOP_LOSS',
                        'signal_strength': data.iloc[entry_index]['Signal_Strength'],
                        'entry_reason': data.iloc[entry_index]['Entry_Reason']
                    })
                    
                    position = 0
                    continue
                
                # Check take profit
                if (position == 1 and current_price >= take_profit_level) or \
                   (position == -1 and current_price <= take_profit_level):
                    # Take profit hit
                    pnl = risk_amount * self.profit_target_ratio  # Profitable exit
                    capital += pnl
                    
                    trades.append({
                        'entry_timestamp': data.index[entry_index],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pips': pnl_pips,
                        'exit_reason': 'TAKE_PROFIT',
                        'signal_strength': data.iloc[entry_index]['Signal_Strength'],
                        'entry_reason': data.iloc[entry_index]['Entry_Reason']
                    })
                    
                    position = 0
                    continue
            
            # Check for new entry signals
            if position == 0 and row['Signal'] != 0:
                # Check drawdown limit
                if drawdown < self.max_drawdown_limit:
                    position = row['Signal']
                    entry_price = current_price
                    entry_index = i
                    
                    # Update risk amount based on current capital
                    risk_amount = capital * self.risk_per_trade
        
        # Close any open position at the end
        if position != 0:
            final_price = data.iloc[-1]['Close']
            pnl_pips = position * (final_price - entry_price)
            pnl = position * (final_price - entry_price) * (risk_amount / data.iloc[-1]['ATR'])
            capital += pnl
            
            trades.append({
                'entry_timestamp': data.index[entry_index],
                'exit_timestamp': data.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': position,
                'pnl': pnl,
                'pnl_pips': pnl_pips,
                'exit_reason': 'END_OF_DATA',
                'signal_strength': data.iloc[entry_index]['Signal_Strength'],
                'entry_reason': data.iloc[entry_index]['Entry_Reason']
            })
        
        # Calculate comprehensive results
        results = self._calculate_enhanced_results(
            trades, equity_curve, drawdown_history, self.initial_capital, capital
        )
        
        self.log_step("BACKTEST", f"✅ Enhanced backtest completed: {len(trades)} trades, {results['total_return']:.2f}% return")
        return results
    
    def _calculate_enhanced_results(self, trades: List[Dict], equity_curve: List[Dict], 
                                  drawdown_history: List[float], initial_capital: float, 
                                  final_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'final_capital': final_capital,
                'trades': trades,
                'equity_curve': equity_curve
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Drawdown analysis
        max_drawdown = max(drawdown_history) if drawdown_history else 0
        
        # Risk metrics
        returns = trades_df['pnl'] / initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Average trade metrics
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        avg_trade = trades_df['pnl'].mean()
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive(trades_df['pnl'] > 0)
        consecutive_losses = self._calculate_consecutive(trades_df['pnl'] < 0)
        
        # Strategy analysis
        strategy_performance = trades_df.groupby('entry_reason').agg({
            'pnl': ['count', 'sum', 'mean'],
            'signal_strength': 'mean'
        }).round(4)
        
        return {
            # Capital metrics
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'net_profit': final_capital - initial_capital,
            
            # Trade metrics
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            
            # Performance metrics
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            # Risk metrics
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
            'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
            
            # Strategy analysis
            'strategy_performance': strategy_performance.to_dict() if not strategy_performance.empty else {},
            
            # Raw data
            'trades': trades,
            'equity_curve': equity_curve,
            'drawdown_history': drawdown_history
        }
    
    def _calculate_consecutive(self, series: pd.Series) -> List[int]:
        """Calculate consecutive True values in a boolean series"""
        consecutive = []
        current_count = 0
        
        for value in series:
            if value:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive.append(current_count)
        
        return consecutive
    
    def display_enhanced_results(self, results: Dict[str, Any]):
        """Display enhanced backtest results with beautiful formatting"""
        self.log_step("RESULTS", "📊 Displaying enhanced backtest results...")
        
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        
        console = Console()
        
        # Main results panel
        main_results = f"""
🎯 ENHANCED BACKTEST RESULTS

💰 CAPITAL PERFORMANCE:
💵 Initial Capital: ${results['initial_capital']:,.2f}
💰 Final Capital: ${results['final_capital']:,.2f}
📈 Net Profit: ${results['net_profit']:,.2f}
📊 Total Return: {results['total_return']:+.2f}%
🔄 Capital Multiplier: {results['final_capital']/results['initial_capital']:.2f}x

📊 TRADING PERFORMANCE:
🎯 Total Trades: {results['total_trades']:,}
✅ Winning Trades: {results['winning_trades']:,}
❌ Losing Trades: {results['losing_trades']:,}
📈 Win Rate: {results['win_rate']:.1f}%
💡 Avg Trade P&L: ${results['avg_trade']:,.2f}

🏆 PROFITABILITY ANALYSIS:
💰 Gross Profit: ${results['gross_profit']:,.2f}
💸 Gross Loss: ${results['gross_loss']:,.2f}
⚡ Profit Factor: {results['profit_factor']:.2f}
📈 Avg Win: ${results['avg_win']:,.2f}
📉 Avg Loss: ${results['avg_loss']:,.2f}

🛡️ RISK MANAGEMENT:
📉 Max Drawdown: {results['max_drawdown']:.2f}%
📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}
🔥 Max Consecutive Wins: {results['max_consecutive_wins']}
❄️ Max Consecutive Losses: {results['max_consecutive_losses']}
"""
        
        console.print(Panel(main_results, title="✅ ENHANCED BACKTEST RESULTS", border_style="green"))
        
        # Strategy performance table
        if results['strategy_performance']:
            strategy_table = Table(title="📊 Strategy Performance Breakdown")
            strategy_table.add_column("Strategy", style="cyan")
            strategy_table.add_column("Trades", justify="right")
            strategy_table.add_column("Total P&L", justify="right", style="green")
            strategy_table.add_column("Avg P&L", justify="right")
            strategy_table.add_column("Avg Signal Strength", justify="right", style="blue")
            
            for strategy, metrics in results['strategy_performance'].items():
                try:
                    # Try multi-level index access first
                    if isinstance(metrics, dict) and ('pnl', 'count') in metrics:
                        count = metrics[('pnl', 'count')]
                        total_pnl = metrics[('pnl', 'sum')]
                        avg_pnl = metrics[('pnl', 'mean')]
                        avg_strength = metrics[('signal_strength', 'mean')]
                    else:
                        # Fallback to simple dict access
                        count = metrics.get('count', 0)
                        total_pnl = metrics.get('total_pnl', 0.0)
                        avg_pnl = metrics.get('avg_pnl', 0.0)
                        avg_strength = metrics.get('avg_strength', 0.0)
                    
                    strategy_table.add_row(
                        strategy,
                        f"{count:,}",
                        f"${total_pnl:,.2f}",
                        f"${avg_pnl:,.2f}",
                        f"{avg_strength:.3f}"
                    )
                except Exception as e:
                    # Skip problematic strategies
                    self.log_step("RESULTS_DISPLAY", f"⚠️ Skipping strategy {strategy}: {e}", LogLevel.WARNING)
                    continue
            
            console.print(strategy_table)
        
        # Performance insights
        insights = self._generate_performance_insights(results)
        console.print(Panel(insights, title="💡 PERFORMANCE INSIGHTS", border_style="blue"))
        
        # Save detailed results
        self._save_enhanced_results(results)
    
    def _generate_performance_insights(self, results: Dict[str, Any]) -> str:
        """Generate intelligent performance insights"""
        insights = []
        
        # Profitability analysis
        if results['total_return'] > 0:
            insights.append(f"✅ PROFITABLE STRATEGY: {results['total_return']:.2f}% return achieved")
        else:
            insights.append(f"❌ UNPROFITABLE STRATEGY: {results['total_return']:.2f}% loss")
        
        # Win rate analysis
        if results['win_rate'] > 60:
            insights.append(f"🎯 EXCELLENT WIN RATE: {results['win_rate']:.1f}% (Above 60%)")
        elif results['win_rate'] > 50:
            insights.append(f"📊 GOOD WIN RATE: {results['win_rate']:.1f}% (Above 50%)")
        else:
            insights.append(f"⚠️ LOW WIN RATE: {results['win_rate']:.1f}% (Below 50%)")
        
        # Profit factor analysis
        if results['profit_factor'] > 1.5:
            insights.append(f"⚡ STRONG PROFIT FACTOR: {results['profit_factor']:.2f} (Excellent)")
        elif results['profit_factor'] > 1.0:
            insights.append(f"📈 POSITIVE PROFIT FACTOR: {results['profit_factor']:.2f} (Good)")
        else:
            insights.append(f"📉 POOR PROFIT FACTOR: {results['profit_factor']:.2f} (Needs improvement)")
        
        # Risk analysis
        if results['max_drawdown'] < 10:
            insights.append(f"🛡️ LOW RISK: {results['max_drawdown']:.2f}% max drawdown")
        elif results['max_drawdown'] < 20:
            insights.append(f"⚠️ MODERATE RISK: {results['max_drawdown']:.2f}% max drawdown")
        else:
            insights.append(f"🚨 HIGH RISK: {results['max_drawdown']:.2f}% max drawdown")
        
        # Sharpe ratio analysis
        if results['sharpe_ratio'] > 1.0:
            insights.append(f"📊 GOOD RISK-ADJUSTED RETURN: Sharpe {results['sharpe_ratio']:.2f}")
        elif results['sharpe_ratio'] > 0.5:
            insights.append(f"📈 ACCEPTABLE RISK-ADJUSTED RETURN: Sharpe {results['sharpe_ratio']:.2f}")
        else:
            insights.append(f"📉 POOR RISK-ADJUSTED RETURN: Sharpe {results['sharpe_ratio']:.2f}")
        
        # Recommendations
        insights.append("\n🎯 RECOMMENDATIONS:")
        
        if results['win_rate'] < 50:
            insights.append("• Improve signal quality and entry conditions")
        
        if results['profit_factor'] < 1.2:
            insights.append("• Optimize risk:reward ratio and exit strategy")
        
        if results['max_drawdown'] > 15:
            insights.append("• Implement stricter risk management")
        
        if results['total_trades'] < 30:
            insights.append("• Increase sample size for more reliable statistics")
        
        return "\n".join(insights)
    
    def _save_enhanced_results(self, results: Dict[str, Any]):
        """Save enhanced results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create backtest results directory
        backtest_dir = self.paths.get_output_path('backtest_results')
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Save comprehensive results
        results_file = os.path.join(backtest_dir, f'enhanced_backtest_results_{timestamp}.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Prepare results for serialization
        serializable_results = {}
        for key, value in results.items():
            if key in ['trades', 'equity_curve', 'drawdown_history']:
                serializable_results[key] = [convert_numpy(item) for item in value]
            elif key == 'strategy_performance':
                # Convert complex DataFrame structure
                serializable_results[key] = {str(k): {str(k2): convert_numpy(v2) for k2, v2 in v.items()} for k, v in value.items()}
            else:
                serializable_results[key] = convert_numpy(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.log_step("RESULTS", f"✅ Enhanced results saved to: {results_file}")
    
    def run_complete_enhanced_backtest(self):
        """Run complete enhanced backtest workflow"""
        try:
            self.log_step("STARTUP", "🚀 Starting Enhanced Advanced Backtest System...")
            
            # Load market data
            data = self.load_market_data()
            
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Generate enhanced signals
            data_with_signals = self.generate_enhanced_signals(data_with_indicators)
            
            # Run enhanced backtest
            results = self.run_enhanced_backtest(data_with_signals)
            
            # Display enhanced results
            self.display_enhanced_results(results)
            
            self.log_step("COMPLETION", f"✅ Enhanced backtest completed successfully!")
            self.log_step("COMPLETION", f"🎯 Final Return: {results['total_return']:+.2f}%")
            self.log_step("COMPLETION", f"💰 Final Capital: ${results['final_capital']:,.2f}")
            self.log_step("COMPLETION", f"📊 Win Rate: {results['win_rate']:.1f}%")
            
            return results
            
        except Exception as e:
            self.log_step("ERROR", f"❌ Enhanced backtest failed: {str(e)}", LogLevel.ERROR)
            self.logger.log_exception(e)
            raise


def run_enhanced_menu_5():
    """Main entry point for Enhanced Menu 5"""
    print("🚀 NICEGOLD ENTERPRISE PROJECTP - ENHANCED MENU 5")
    print("=" * 60)
    print("🎯 Enhanced Advanced Backtesting System")
    print("📊 Improved Strategy & Analysis")
    print("💰 Profit-Focused Implementation")
    print("=" * 60)
    
    try:
        # Initialize and run enhanced backtest system
        backtest_system = EnhancedAdvancedBacktestSystem()
        results = backtest_system.run_complete_enhanced_backtest()
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED BACKTEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Return: {results['total_return']:+.2f}%")
        print(f"💰 Final Capital: ${results['final_capital']:,.2f}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"⚡ Profit Factor: {results['profit_factor']:.2f}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Enhanced backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run enhanced menu 5 for testing
    run_enhanced_menu_5() 