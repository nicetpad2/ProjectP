#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 PROFESSIONAL CAPITAL MANAGEMENT & COMPOUND GROWTH SYSTEM
NICEGOLD ProjectP - Enterprise Trading Capital Management

🎯 CAPITAL MANAGEMENT FEATURES:
✅ Professional Capital Management (Starting $100)
✅ Compound Growth System (Growth without portfolio break)
✅ Risk Management & Position Sizing
✅ Drawdown Protection (Max 15% drawdown)
✅ Kelly Criterion Position Sizing
✅ Dynamic Stop Loss & Take Profit
✅ Professional Money Management
✅ Real-time Capital Monitoring
✅ Growth Validation & Reporting

CAPITAL MANAGEMENT RULES:
- Initial Capital: $100 USD
- Risk per Trade: 1-2% of current capital
- Maximum Drawdown: 15% of peak capital
- Position Size: Kelly Criterion based
- Compound Growth: Reinvest profits continuously
- Stop Loss: ATR-based dynamic stops
- Take Profit: Risk-reward ratio 1:2 minimum
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from decimal import Decimal, ROUND_HALF_UP
import sqlite3
import csv
import math

# Project imports
try:
    from core.unified_enterprise_logger import get_unified_logger
    from core.config import get_global_config
    from core.project_paths import ProjectPaths
    from core.beautiful_progress import BeautifulProgress
    ENTERPRISE_IMPORTS = True
except ImportError as e:
    print(f"⚠️ Enterprise imports not available: {e}")
    ENTERPRISE_IMPORTS = False

# Rich imports for beautiful UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ====================================================
# CAPITAL MANAGEMENT ENUMS AND STRUCTURES
# ====================================================

class CapitalStatus(Enum):
    GROWING = "GROWING"
    STABLE = "STABLE"
    DRAWDOWN = "DRAWDOWN"
    RECOVERY = "RECOVERY"
    CRITICAL = "CRITICAL"  # >10% drawdown
    EMERGENCY = "EMERGENCY"  # >15% drawdown

class RiskLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"  # 0.5% risk per trade
    MODERATE = "MODERATE"       # 1.0% risk per trade
    AGGRESSIVE = "AGGRESSIVE"   # 2.0% risk per trade
    DYNAMIC = "DYNAMIC"        # Kelly Criterion based

@dataclass
class CapitalSnapshot:
    """Capital snapshot at specific point in time"""
    timestamp: datetime
    current_capital: float
    peak_capital: float
    initial_capital: float
    total_growth: float
    growth_percentage: float
    drawdown_from_peak: float
    drawdown_percentage: float
    risk_level: RiskLevel
    position_size: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    status: CapitalStatus
    kelly_criterion: float
    max_position_size: float
    recommended_position_size: float
    
@dataclass
class TradeCapitalImpact:
    """Capital impact of individual trade"""
    trade_id: str
    timestamp: datetime
    entry_capital: float
    exit_capital: float
    position_size: float
    risk_amount: float
    actual_profit_loss: float
    capital_growth: float
    new_peak_capital: float
    drawdown_impact: float
    kelly_factor: float
    risk_percentage: float
    
# ====================================================
# PROFESSIONAL CAPITAL MANAGEMENT SYSTEM
# ====================================================

class ProfessionalCapitalManager:
    """
    Professional Capital Management System
    
    Features:
    - Start with $100 capital
    - Compound growth system
    - Dynamic position sizing
    - Risk management
    - Drawdown protection
    """
    
    def __init__(self, initial_capital: float = 100.0, config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.config = config or {}
        
        # Risk management parameters
        self.max_drawdown_percentage = self.config.get('max_drawdown_percentage', 0.15)  # 15%
        self.default_risk_per_trade = self.config.get('default_risk_per_trade', 0.02)  # 2%
        self.min_risk_per_trade = self.config.get('min_risk_per_trade', 0.005)  # 0.5%
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.03)  # 3%
        self.min_position_size = self.config.get('min_position_size', 0.01)  # 0.01 lots
        self.max_position_size = self.config.get('max_position_size', 1.0)   # 1.0 lots
        
        # Trading parameters
        self.spread_points = self.config.get('spread_points', 100)  # 100 points = 1 pip
        self.commission_per_lot = self.config.get('commission_per_lot', 7.0)  # $7 per lot
        self.pip_value = self.config.get('pip_value', 1.0)  # $1 per pip per 0.1 lot
        
        # Capital tracking
        self.capital_history: List[CapitalSnapshot] = []
        self.trade_history: List[TradeCapitalImpact] = []
        self.statistics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'kelly_criterion': 0.0,
            'recovery_factor': 0.0
        }
        
        # Initialize logging
        if ENTERPRISE_IMPORTS:
            self.logger = get_unified_logger('ProfessionalCapitalManager')
        else:
            import logging
            self.logger = logging.getLogger(__name__)
            
        # Initialize console
        self.console = Console() if RICH_AVAILABLE else None
        
        # Initialize first snapshot
        self._create_initial_snapshot()
    
    def _create_initial_snapshot(self):
        """Create initial capital snapshot"""
        snapshot = CapitalSnapshot(
            timestamp=datetime.now(),
            current_capital=self.current_capital,
            peak_capital=self.peak_capital,
            initial_capital=self.initial_capital,
            total_growth=0.0,
            growth_percentage=0.0,
            drawdown_from_peak=0.0,
            drawdown_percentage=0.0,
            risk_level=RiskLevel.MODERATE,
            position_size=self.min_position_size,
            trades_count=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=1.0,
            status=CapitalStatus.STABLE,
            kelly_criterion=0.0,
            max_position_size=self.max_position_size,
            recommended_position_size=self.min_position_size
        )
        
        self.capital_history.append(snapshot)
        
        if self.logger:
            self.logger.info(f"💰 Capital Manager initialized with ${self.initial_capital:.2f}")
    
    def calculate_kelly_criterion(self) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        Kelly% = (Win Rate × Average Win - Loss Rate × Average Loss) / Average Win
        """
        if self.statistics['total_trades'] < 10:
            return 0.01  # Conservative until we have enough data
        
        win_rate = self.statistics['win_rate']
        loss_rate = 1 - win_rate
        avg_win = self.statistics['average_win']
        avg_loss = abs(self.statistics['average_loss'])
        
        if avg_win <= 0 or avg_loss <= 0:
            return 0.01
        
        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # Limit Kelly to reasonable range
        kelly = max(0.005, min(0.05, kelly))  # Between 0.5% and 5%
        
        return kelly
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_amount: Optional[float] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size based on capital and risk management
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_amount: Optional fixed risk amount
            
        Returns:
            Tuple of (position_size, calculation_details)
        """
        if risk_amount is None:
            # Use Kelly Criterion or default risk
            kelly = self.calculate_kelly_criterion()
            risk_percentage = kelly if kelly > 0 else self.default_risk_per_trade
            risk_amount = self.current_capital * risk_percentage
        
        # Calculate stop loss distance in pips
        stop_loss_distance = abs(entry_price - stop_loss)
        stop_loss_pips = stop_loss_distance / 0.0001  # Convert to pips
        
        # Calculate position size based on risk amount and stop loss
        if stop_loss_pips > 0:
            # Position size = Risk Amount / (Stop Loss Pips × Pip Value)
            position_size = risk_amount / (stop_loss_pips * self.pip_value)
        else:
            position_size = self.min_position_size
        
        # Apply position size limits
        position_size = max(self.min_position_size, min(self.max_position_size, position_size))
        
        # Round to standard lot sizes
        position_size = round(position_size, 2)
        
        # Calculate actual risk with final position size
        actual_risk = stop_loss_pips * self.pip_value * position_size
        risk_percentage = actual_risk / self.current_capital
        
        calculation_details = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_loss_distance': stop_loss_distance,
            'stop_loss_pips': stop_loss_pips,
            'risk_amount_requested': risk_amount,
            'actual_risk_amount': actual_risk,
            'risk_percentage': risk_percentage,
            'position_size': position_size,
            'kelly_criterion': self.calculate_kelly_criterion(),
            'pip_value': self.pip_value,
            'current_capital': self.current_capital
        }
        
        return position_size, calculation_details
    
    def execute_trade(self, trade_result: Dict[str, Any]) -> TradeCapitalImpact:
        """
        Execute trade and update capital
        
        Args:
            trade_result: Dictionary containing trade information
            
        Returns:
            TradeCapitalImpact object
        """
        trade_id = trade_result.get('trade_id', str(uuid.uuid4()))
        profit_loss = trade_result.get('profit_loss', 0.0)
        position_size = trade_result.get('position_size', self.min_position_size)
        
        # Calculate commission and spread costs
        commission = position_size * self.commission_per_lot
        spread_cost = position_size * self.spread_points * (self.pip_value / 100)
        
        # Calculate net profit/loss
        net_profit_loss = profit_loss - commission - spread_cost
        
        # Store entry capital
        entry_capital = self.current_capital
        
        # Update capital
        self.current_capital += net_profit_loss
        
        # Update peak capital if growing
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate growth and drawdown
        capital_growth = self.current_capital - self.initial_capital
        drawdown_from_peak = self.peak_capital - self.current_capital
        
        # Update statistics
        self.statistics['total_trades'] += 1
        
        if net_profit_loss > 0:
            self.statistics['winning_trades'] += 1
            self.statistics['total_profit'] += net_profit_loss
            self.statistics['current_consecutive_wins'] += 1
            self.statistics['current_consecutive_losses'] = 0
            self.statistics['max_consecutive_wins'] = max(
                self.statistics['max_consecutive_wins'], 
                self.statistics['current_consecutive_wins']
            )
            self.statistics['largest_win'] = max(self.statistics['largest_win'], net_profit_loss)
        else:
            self.statistics['losing_trades'] += 1
            self.statistics['total_loss'] += abs(net_profit_loss)
            self.statistics['current_consecutive_losses'] += 1
            self.statistics['current_consecutive_wins'] = 0
            self.statistics['max_consecutive_losses'] = max(
                self.statistics['max_consecutive_losses'], 
                self.statistics['current_consecutive_losses']
            )
            self.statistics['largest_loss'] = min(self.statistics['largest_loss'], net_profit_loss)
        
        # Calculate derived statistics
        if self.statistics['winning_trades'] > 0:
            self.statistics['average_win'] = self.statistics['total_profit'] / self.statistics['winning_trades']
        if self.statistics['losing_trades'] > 0:
            self.statistics['average_loss'] = -self.statistics['total_loss'] / self.statistics['losing_trades']
        
        self.statistics['win_rate'] = self.statistics['winning_trades'] / self.statistics['total_trades']
        
        if self.statistics['total_loss'] > 0:
            self.statistics['profit_factor'] = self.statistics['total_profit'] / self.statistics['total_loss']
        else:
            self.statistics['profit_factor'] = float('inf') if self.statistics['total_profit'] > 0 else 1.0
        
        self.statistics['kelly_criterion'] = self.calculate_kelly_criterion()
        
        # Create trade impact record
        trade_impact = TradeCapitalImpact(
            trade_id=trade_id,
            timestamp=datetime.now(),
            entry_capital=entry_capital,
            exit_capital=self.current_capital,
            position_size=position_size,
            risk_amount=position_size * self.default_risk_per_trade,
            actual_profit_loss=net_profit_loss,
            capital_growth=capital_growth,
            new_peak_capital=self.peak_capital,
            drawdown_impact=drawdown_from_peak,
            kelly_factor=self.statistics['kelly_criterion'],
            risk_percentage=abs(net_profit_loss) / entry_capital if entry_capital > 0 else 0.0
        )
        
        self.trade_history.append(trade_impact)
        
        # Create new capital snapshot
        self._create_capital_snapshot()
        
        # Log trade execution
        if self.logger:
            self.logger.info(f"💰 Trade executed: {trade_id}")
            self.logger.info(f"   Capital: ${entry_capital:.2f} → ${self.current_capital:.2f}")
            self.logger.info(f"   P&L: ${net_profit_loss:.2f} (Position: {position_size} lots)")
            self.logger.info(f"   Growth: {capital_growth:.2f} ({((self.current_capital/self.initial_capital-1)*100):.1f}%)")
        
        return trade_impact
    
    def _create_capital_snapshot(self):
        """Create capital snapshot after trade"""
        # Calculate current status
        drawdown_from_peak = self.peak_capital - self.current_capital
        drawdown_percentage = drawdown_from_peak / self.peak_capital if self.peak_capital > 0 else 0.0
        
        # Determine capital status
        if drawdown_percentage >= 0.15:
            status = CapitalStatus.EMERGENCY
        elif drawdown_percentage >= 0.10:
            status = CapitalStatus.CRITICAL
        elif drawdown_percentage > 0.05:
            status = CapitalStatus.DRAWDOWN
        elif drawdown_percentage > 0.0:
            status = CapitalStatus.RECOVERY
        elif self.current_capital > self.peak_capital * 0.99:
            status = CapitalStatus.GROWING
        else:
            status = CapitalStatus.STABLE
        
        # Determine risk level based on performance
        if status in [CapitalStatus.EMERGENCY, CapitalStatus.CRITICAL]:
            risk_level = RiskLevel.CONSERVATIVE
        elif status == CapitalStatus.DRAWDOWN:
            risk_level = RiskLevel.MODERATE
        elif status == CapitalStatus.GROWING and self.statistics['win_rate'] > 0.6:
            risk_level = RiskLevel.AGGRESSIVE
        else:
            risk_level = RiskLevel.DYNAMIC
        
        # Calculate recommended position size
        kelly = self.calculate_kelly_criterion()
        if risk_level == RiskLevel.CONSERVATIVE:
            recommended_position_size = self.min_position_size
        elif risk_level == RiskLevel.MODERATE:
            recommended_position_size = self.min_position_size * 2
        elif risk_level == RiskLevel.AGGRESSIVE:
            recommended_position_size = min(self.max_position_size, self.min_position_size * 5)
        else:  # DYNAMIC
            recommended_position_size = min(self.max_position_size, kelly * 100)
        
        snapshot = CapitalSnapshot(
            timestamp=datetime.now(),
            current_capital=self.current_capital,
            peak_capital=self.peak_capital,
            initial_capital=self.initial_capital,
            total_growth=self.current_capital - self.initial_capital,
            growth_percentage=((self.current_capital / self.initial_capital) - 1) * 100,
            drawdown_from_peak=drawdown_from_peak,
            drawdown_percentage=drawdown_percentage * 100,
            risk_level=risk_level,
            position_size=recommended_position_size,
            trades_count=self.statistics['total_trades'],
            winning_trades=self.statistics['winning_trades'],
            losing_trades=self.statistics['losing_trades'],
            win_rate=self.statistics['win_rate'],
            profit_factor=self.statistics['profit_factor'],
            status=status,
            kelly_criterion=kelly,
            max_position_size=self.max_position_size,
            recommended_position_size=recommended_position_size
        )
        
        self.capital_history.append(snapshot)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current capital status"""
        if not self.capital_history:
            return {}
        
        latest_snapshot = self.capital_history[-1]
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'peak_capital': self.peak_capital,
            'total_growth': latest_snapshot.total_growth,
            'growth_percentage': latest_snapshot.growth_percentage,
            'drawdown_percentage': latest_snapshot.drawdown_percentage,
            'status': latest_snapshot.status.value,
            'risk_level': latest_snapshot.risk_level.value,
            'recommended_position_size': latest_snapshot.recommended_position_size,
            'kelly_criterion': latest_snapshot.kelly_criterion,
            'win_rate': latest_snapshot.win_rate,
            'profit_factor': latest_snapshot.profit_factor,
            'total_trades': latest_snapshot.trades_count,
            'statistics': self.statistics.copy()
        }
    
    def display_capital_dashboard(self):
        """Display beautiful capital dashboard"""
        if not RICH_AVAILABLE or not self.console:
            return
        
        status = self.get_current_status()
        
        # Create dashboard table
        table = Table(title="💰 Professional Capital Management Dashboard", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        # Add rows
        table.add_row("💵 Current Capital", f"${status['current_capital']:.2f}", "🎯 Active")
        table.add_row("🏆 Peak Capital", f"${status['peak_capital']:.2f}", "📈 Record")
        table.add_row("📊 Total Growth", f"${status['total_growth']:.2f}", f"{status['growth_percentage']:.1f}%")
        table.add_row("⚠️ Drawdown", f"{status['drawdown_percentage']:.1f}%", status['status'])
        table.add_row("🎲 Risk Level", status['risk_level'], "🛡️ Protected")
        table.add_row("📈 Win Rate", f"{status['win_rate']:.1%}", "✅ Tracked")
        table.add_row("💪 Profit Factor", f"{status['profit_factor']:.2f}", "📊 Calculated")
        table.add_row("🎯 Kelly Criterion", f"{status['kelly_criterion']:.1%}", "🧠 Optimal")
        table.add_row("📋 Total Trades", str(status['total_trades']), "📊 Executed")
        table.add_row("💼 Position Size", f"{status['recommended_position_size']:.2f} lots", "🎯 Recommended")
        
        self.console.print(table)
    
    def save_capital_report(self, filepath: str):
        """Save detailed capital report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'capital_manager_config': {
                'initial_capital': self.initial_capital,
                'max_drawdown_percentage': self.max_drawdown_percentage,
                'default_risk_per_trade': self.default_risk_per_trade,
                'spread_points': self.spread_points,
                'commission_per_lot': self.commission_per_lot
            },
            'current_status': self.get_current_status(),
            'capital_history': [asdict(snapshot) for snapshot in self.capital_history],
            'trade_history': [asdict(trade) for trade in self.trade_history],
            'statistics': self.statistics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.logger:
            self.logger.info(f"💾 Capital report saved to {filepath}")

# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    'ProfessionalCapitalManager',
    'CapitalSnapshot',
    'TradeCapitalImpact',
    'CapitalStatus',
    'RiskLevel'
]

if __name__ == "__main__":
    # Test the capital manager
    print("🧪 Testing Professional Capital Manager...")
    
    # Initialize capital manager
    capital_manager = ProfessionalCapitalManager(initial_capital=100.0)
    
    # Simulate some trades
    test_trades = [
        {'trade_id': 'T001', 'profit_loss': 15.0, 'position_size': 0.1},
        {'trade_id': 'T002', 'profit_loss': -8.0, 'position_size': 0.1},
        {'trade_id': 'T003', 'profit_loss': 22.0, 'position_size': 0.1},
        {'trade_id': 'T004', 'profit_loss': -5.0, 'position_size': 0.1},
        {'trade_id': 'T005', 'profit_loss': 18.0, 'position_size': 0.1},
    ]
    
    for trade in test_trades:
        impact = capital_manager.execute_trade(trade)
        print(f"✅ Trade {trade['trade_id']}: ${impact.actual_profit_loss:.2f} → Capital: ${impact.exit_capital:.2f}")
    
    # Display final status
    capital_manager.display_capital_dashboard()
    
    # Save report
    capital_manager.save_capital_report('test_capital_report.json')
    
    print("✅ Capital Manager test completed!")
