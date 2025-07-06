#!/usr/bin/env python3
"""
🤖 ENHANCED MULTI-TIMEFRAME DQN AGENT
Enhanced Deep Q-Network Agent with Multi-Timeframe Elliott Wave Integration

Advanced Features:
- Multi-Timeframe State Representation
- Elliott Wave Pattern Recognition Integration
- Advanced Reward Function with Wave Analysis
- Hierarchical Learning Architecture
- Market Regime Awareness
- Dynamic Action Space
- Advanced Experience Replay with Prioritization
"""

import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
import json
import sys
from pathlib import Path
from collections import deque
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Check PyTorch availability
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

class EnhancedDQNNetwork(nn.Module):
    """Enhanced Deep Q-Network with Multi-Timeframe and Elliott Wave Integration"""
    
    def __init__(self, state_size: int, action_size: int, 
                 timeframe_features: int = 7, elliott_features: int = 20):
        super(EnhancedDQNNetwork, self).__init__()
        
        # Multi-Timeframe Feature Extractor
        self.timeframe_encoder = nn.Sequential(
            nn.Linear(timeframe_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Elliott Wave Feature Extractor
        self.elliott_encoder = nn.Sequential(
            nn.Linear(elliott_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Technical Features Extractor
        technical_size = state_size - timeframe_features - elliott_features
        self.technical_encoder = nn.Sequential(
            nn.Linear(technical_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Feature Fusion Layer
        combined_size = 64 + 64 + 128  # timeframe + elliott + technical
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Market Regime Classifier (auxiliary task)
        self.regime_classifier = nn.Linear(256, 4)  # Trend/Range/Volatile/Calm
        
        # Action Value Estimator
        self.action_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x, return_regime=False):
        """Forward pass with optional regime prediction"""
        # Split input into different feature types
        timeframe_features = x[:, :7]  # First 7 features
        elliott_features = x[:, 7:27]   # Next 20 features
        technical_features = x[:, 27:]  # Remaining features
        
        # Encode each feature type
        tf_encoded = self.timeframe_encoder(timeframe_features)
        elliott_encoded = self.elliott_encoder(elliott_features)
        tech_encoded = self.technical_encoder(technical_features)
        
        # Combine features
        combined = torch.cat([tf_encoded, elliott_encoded, tech_encoded], dim=1)
        fused = self.fusion_layer(combined)
        
        # Predict actions
        q_values = self.action_layers(fused)
        
        if return_regime:
            regime_pred = self.regime_classifier(fused)
            return q_values, regime_pred
        
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience with priority"""
        if priority is None:
            priority = max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(None)
            self.size += 1
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample experiences with prioritized sampling"""
        if self.size < batch_size:
            return None
            
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class EnhancedMultiTimeframeDQNAgent:
    """Enhanced DQN Agent with Multi-Timeframe Elliott Wave Integration"""
    
    def __init__(self, state_size: int = 50, action_size: int = 3, learning_rate: float = 0.001):
        """Initialize Enhanced Multi-Timeframe DQN Agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        # Enhanced Hyperparameters
        self.gamma = 0.99  # Increased discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64  # Increased batch size
        self.target_update_freq = 100
        self.memory_size = 50000  # Increased memory
        
        # Multi-Timeframe Parameters
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        self.timeframe_weights = {
            'M1': 0.1, 'M5': 0.15, 'M15': 0.2, 'M30': 0.2,
            'H1': 0.15, 'H4': 0.15, 'D1': 0.05
        }
        
        # Elliott Wave Integration
        self.elliott_wave_analyzer = None
        self.wave_awareness = True
        
        # Advanced Reward Parameters
        self.reward_components = {
            'profit': 1.0,
            'wave_alignment': 0.3,
            'risk_management': 0.2,
            'timeframe_confluence': 0.25,
            'drawdown_penalty': -0.5
        }
        
        # Initialize PyTorch availability flag FIRST
        self.pytorch_available = PYTORCH_AVAILABLE
        
        # Initialize Networks (if PyTorch available)
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cpu")  # Force CPU
            self.q_network = EnhancedDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = EnhancedDQNNetwork(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.update_target_network()
            
            # Prioritized Replay Buffer
            self.memory = PrioritizedReplayBuffer(self.memory_size)
            
            self.logger.info("✅ Enhanced PyTorch DQN networks initialized")
        else:
            self.logger.warning("⚠️ PyTorch not available - using fallback agent")
            self._init_fallback_agent()
        
        # Trading State
        self.reset_trading_state()
        
        # Performance Tracking
        self.episode_rewards = []
        self.episode_trades = []
        self.training_metrics = {
            'total_episodes': 0,
            'total_reward': 0.0,
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def reset_trading_state(self):
        """Reset trading state for new episode"""
        self.position = 0  # 0=neutral, 1=long, -1=short
        self.entry_price = 0.0
        self.portfolio_value = 10000.0  # Starting capital
        self.max_portfolio_value = self.portfolio_value
        self.trade_history = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
    def _init_fallback_agent(self):
        """Initialize fallback agent when PyTorch not available"""
        self.q_table = np.random.randn(100, self.action_size) * 0.1
        self.state_history = deque(maxlen=100)
        
    def update_target_network(self):
        """Update target network with current network weights"""
        if self.pytorch_available:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def encode_state(self, market_data: Dict[str, Any], 
                    elliott_analysis: Dict[str, Any] = None) -> np.ndarray:
        """Encode market state with multi-timeframe and Elliott Wave features"""
        try:
            state_features = []
            
            # Multi-Timeframe Features (7 features)
            timeframe_features = self._extract_timeframe_features(market_data)
            state_features.extend(timeframe_features)
            
            # Elliott Wave Features (20 features)
            elliott_features = self._extract_elliott_wave_features(elliott_analysis)
            state_features.extend(elliott_features)
            
            # Technical Features (remaining)
            technical_features = self._extract_technical_features(market_data)
            state_features.extend(technical_features)
            
            # Ensure correct state size
            while len(state_features) < self.state_size:
                state_features.append(0.0)
            
            state_array = np.array(state_features[:self.state_size], dtype=np.float32)
            
            # Handle NaN/inf values
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state_array
            
        except Exception as e:
            self.logger.error(f"State encoding failed: {str(e)}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def _extract_timeframe_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract multi-timeframe features"""
        try:
            features = []
            
            # Get timeframe data if available
            timeframe_data = market_data.get('timeframe_data', {})
            
            for tf in self.timeframes:
                if tf in timeframe_data:
                    tf_data = timeframe_data[tf]
                    if len(tf_data) > 0:
                        # Calculate trend strength for this timeframe
                        close_prices = tf_data['close'].tail(20)
                        if len(close_prices) >= 2:
                            trend = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
                            features.append(np.tanh(trend * 100))  # Normalize to [-1, 1]
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Timeframe features extraction failed: {str(e)}")
            return [0.0] * len(self.timeframes)
    
    def _extract_elliott_wave_features(self, elliott_analysis: Dict[str, Any] = None) -> List[float]:
        """Extract Elliott Wave features"""
        try:
            features = [0.0] * 20  # 20 Elliott Wave features
            
            if not elliott_analysis:
                return features
            
            # Feature 0-2: Wave Type and Direction
            confluence = elliott_analysis.get('confluence_analysis', {})
            direction = confluence.get('overall_direction', 'NEUTRAL')
            
            if direction == 'UPTREND':
                features[0] = 1.0
            elif direction == 'DOWNTREND':
                features[0] = -1.0
            else:
                features[0] = 0.0
            
            # Feature 1: Wave Strength
            features[1] = confluence.get('strength', 0) / 10.0  # Normalize to [0, 1]
            
            # Feature 2: Confluence Agreement
            features[2] = confluence.get('agreement_score', 0.0)
            
            # Features 3-7: Timeframe Agreement
            supporting_tfs = confluence.get('supporting_timeframes', [])
            for i, tf in enumerate(self.timeframes[:5]):  # First 5 timeframes
                features[3 + i] = 1.0 if tf in supporting_tfs else 0.0
            
            # Features 8-12: Wave Count Information
            primary_wave = elliott_analysis.get('primary_wave_count')
            if primary_wave:
                if '5-WAVE' in primary_wave:
                    features[8] = 1.0
                elif '3-WAVE' in primary_wave:
                    features[9] = 1.0
                
                # Current wave position
                if 'WAVE_1' in str(primary_wave):
                    features[10] = 0.2
                elif 'WAVE_2' in str(primary_wave):
                    features[10] = 0.4
                elif 'WAVE_3' in str(primary_wave):
                    features[10] = 0.6
                elif 'WAVE_4' in str(primary_wave):
                    features[10] = 0.8
                elif 'WAVE_5' in str(primary_wave):
                    features[10] = 1.0
            
            # Features 13-17: Fibonacci Level Proximity
            fibonacci_levels = elliott_analysis.get('fibonacci_levels', {})
            key_levels = fibonacci_levels.get('key_levels', [])
            for i, level in enumerate(key_levels[:5]):
                if level['distance_percent'] < 2.0:  # Within 2%
                    features[13 + i] = 1.0 - (level['distance_percent'] / 2.0)
            
            # Features 18-19: Trading Signals
            signals = elliott_analysis.get('trading_signals', [])
            buy_signal_strength = 0.0
            sell_signal_strength = 0.0
            
            for signal in signals:
                if signal['type'] == 'BUY':
                    buy_signal_strength = max(buy_signal_strength, signal['strength'] / 10.0)
                elif signal['type'] == 'SELL':
                    sell_signal_strength = max(sell_signal_strength, signal['strength'] / 10.0)
            
            features[18] = buy_signal_strength
            features[19] = sell_signal_strength
            
            return features
            
        except Exception as e:
            self.logger.error(f"Elliott Wave features extraction failed: {str(e)}")
            return [0.0] * 20
    
    def _extract_technical_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract technical analysis features"""
        try:
            features = []
            
            # Price features
            current_price = market_data.get('close', 0)
            if current_price > 0:
                # RSI
                rsi = market_data.get('rsi', 50) / 100.0  # Normalize to [0, 1]
                features.append(rsi)
                
                # MACD
                macd = market_data.get('macd', 0)
                macd_signal = market_data.get('macd_signal', 0)
                macd_diff = (macd - macd_signal) / current_price if current_price > 0 else 0
                features.append(np.tanh(macd_diff * 1000))  # Normalize
                
                # Bollinger Bands
                bb_upper = market_data.get('bb_upper', current_price)
                bb_lower = market_data.get('bb_lower', current_price)
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                features.append(bb_position)
                
                # Volume
                volume = market_data.get('volume', 0)
                avg_volume = market_data.get('avg_volume', volume)
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                features.append(min(volume_ratio, 3.0) / 3.0)  # Cap at 3x and normalize
                
                # Volatility
                volatility = market_data.get('volatility', 0)
                features.append(min(volatility, 0.1) / 0.1)  # Normalize volatility
                
            else:
                features.extend([0.5, 0.0, 0.5, 1.0, 0.5])  # Default values
            
            # Extend to fill remaining space
            remaining_features = self.state_size - 27  # 7 timeframe + 20 elliott
            while len(features) < remaining_features:
                features.append(0.0)
            
            return features[:remaining_features]
            
        except Exception as e:
            self.logger.error(f"Technical features extraction failed: {str(e)}")
            remaining_features = self.state_size - 27
            return [0.0] * remaining_features
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy with enhanced logic"""
        try:
            if training and random.random() <= self.epsilon:
                return random.choice(range(self.action_size))
            
            if self.pytorch_available:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values, regime_pred = self.q_network(state_tensor, return_regime=True)
                    
                    # Get base action from Q-values
                    base_action = q_values.max(1)[1].item()
                    
                    # Apply Elliott Wave and multi-timeframe logic
                    adjusted_action = self._apply_enhanced_action_logic(
                        base_action, state, q_values.squeeze().cpu().numpy()
                    )
                    
                    return adjusted_action
            else:
                # Fallback logic
                return self._fallback_action(state)
                
        except Exception as e:
            self.logger.error(f"Action selection failed: {str(e)}")
            return 1  # Default to hold
    
    def _apply_enhanced_action_logic(self, base_action: int, state: np.ndarray, 
                                   q_values: np.ndarray) -> int:
        """Apply enhanced action logic considering Elliott Wave and multi-timeframe analysis"""
        try:
            # Extract Elliott Wave signals from state
            elliott_features = state[7:27]
            
            # Wave direction (feature 0)
            wave_direction = elliott_features[0]
            
            # Wave strength (feature 1)
            wave_strength = elliott_features[1]
            
            # Buy/Sell signal strength (features 18-19)
            buy_signal = elliott_features[18]
            sell_signal = elliott_features[19]
            
            # Timeframe confluence
            timeframe_agreement = np.mean(elliott_features[3:8])  # Features 3-7
            
            # Apply logic
            if wave_strength > 0.7 and timeframe_agreement > 0.6:
                # Strong wave with good timeframe agreement
                if wave_direction > 0.5 and buy_signal > 0.5:
                    return 0  # Strong buy signal
                elif wave_direction < -0.5 and sell_signal > 0.5:
                    return 2  # Strong sell signal
            
            # Risk management - avoid weak signals
            if wave_strength < 0.3 or timeframe_agreement < 0.3:
                return 1  # Hold in uncertain conditions
            
            # Use Q-values if no strong Elliott Wave signal
            return base_action
            
        except Exception as e:
            self.logger.error(f"Enhanced action logic failed: {str(e)}")
            return base_action
    
    def _fallback_action(self, state: np.ndarray) -> int:
        """Fallback action selection when PyTorch not available"""
        try:
            # Simple rule-based logic using state features
            # Use first few features for decision making
            if len(state) >= 3:
                trend_signal = state[0]  # First timeframe feature
                elliott_direction = state[7] if len(state) > 7 else 0  # Elliott direction
                
                if trend_signal > 0.1 and elliott_direction > 0.1:
                    return 0  # Buy
                elif trend_signal < -0.1 and elliott_direction < -0.1:
                    return 2  # Sell
                else:
                    return 1  # Hold
            
            return 1  # Default hold
            
        except Exception as e:
            self.logger.error(f"Fallback action failed: {str(e)}")
            return 1
    
    def calculate_enhanced_reward(self, action: int, market_data: Dict[str, Any],
                                elliott_analysis: Dict[str, Any] = None) -> float:
        """Calculate enhanced reward considering multiple factors"""
        try:
            total_reward = 0.0
            reward_breakdown = {}
            
            current_price = market_data.get('close', 0)
            if current_price <= 0:
                return 0.0
            
            # 1. Basic Profit/Loss Reward
            profit_reward = self._calculate_profit_reward(action, current_price)
            total_reward += profit_reward * self.reward_components['profit']
            reward_breakdown['profit'] = profit_reward
            
            # 2. Elliott Wave Alignment Reward
            if elliott_analysis:
                wave_reward = self._calculate_wave_alignment_reward(action, elliott_analysis)
                total_reward += wave_reward * self.reward_components['wave_alignment']
                reward_breakdown['wave_alignment'] = wave_reward
            
            # 3. Risk Management Reward
            risk_reward = self._calculate_risk_management_reward(action, current_price)
            total_reward += risk_reward * self.reward_components['risk_management']
            reward_breakdown['risk_management'] = risk_reward
            
            # 4. Timeframe Confluence Reward
            if elliott_analysis:
                confluence_reward = self._calculate_confluence_reward(action, elliott_analysis)
                total_reward += confluence_reward * self.reward_components['timeframe_confluence']
                reward_breakdown['confluence'] = confluence_reward
            
            # 5. Drawdown Penalty
            drawdown_penalty = self._calculate_drawdown_penalty()
            total_reward += drawdown_penalty * self.reward_components['drawdown_penalty']
            reward_breakdown['drawdown'] = drawdown_penalty
            
            # Log reward breakdown for analysis
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Reward breakdown: {reward_breakdown}, Total: {total_reward:.4f}")
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"Enhanced reward calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_profit_reward(self, action: int, current_price: float) -> float:
        """Calculate profit-based reward"""
        try:
            if self.position == 0:  # No position
                if action != 1:  # Opening position
                    self.position = 1 if action == 0 else -1
                    self.entry_price = current_price
                return 0.0
            
            # Calculate unrealized P&L
            if self.position == 1:  # Long position
                pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                pnl = (self.entry_price - current_price) / self.entry_price
            
            if action == 1:  # Closing position
                self.position = 0
                self.entry_price = 0.0
                return pnl * 100  # Scale reward
            
            return pnl * 10  # Smaller reward for unrealized P&L
            
        except Exception as e:
            self.logger.error(f"Profit reward calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_wave_alignment_reward(self, action: int, 
                                       elliott_analysis: Dict[str, Any]) -> float:
        """Calculate reward for Elliott Wave alignment"""
        try:
            confluence = elliott_analysis.get('confluence_analysis', {})
            direction = confluence.get('overall_direction', 'NEUTRAL')
            strength = confluence.get('strength', 0) / 10.0
            
            # Reward for aligning with wave direction
            if direction == 'UPTREND' and action == 0:  # Buy in uptrend
                return strength * 2.0
            elif direction == 'DOWNTREND' and action == 2:  # Sell in downtrend
                return strength * 2.0
            elif direction in ['SIDEWAYS', 'NEUTRAL'] and action == 1:  # Hold in sideways
                return strength * 1.0
            else:
                return -strength * 1.0  # Penalty for going against trend
                
        except Exception as e:
            self.logger.error(f"Wave alignment reward calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_risk_management_reward(self, action: int, current_price: float) -> float:
        """Calculate risk management reward"""
        try:
            reward = 0.0
            
            # Reward for not overtrading
            if action == 1:  # Hold action
                reward += 0.1
            
            # Penalty for large positions without stop loss
            if self.position != 0 and self.entry_price > 0:
                unrealized_loss = 0
                if self.position == 1:  # Long
                    unrealized_loss = (self.entry_price - current_price) / self.entry_price
                else:  # Short
                    unrealized_loss = (current_price - self.entry_price) / self.entry_price
                
                if unrealized_loss > 0.02:  # More than 2% loss
                    reward -= unrealized_loss * 5.0
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Risk management reward calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_confluence_reward(self, action: int, 
                                   elliott_analysis: Dict[str, Any]) -> float:
        """Calculate timeframe confluence reward"""
        try:
            confluence = elliott_analysis.get('confluence_analysis', {})
            agreement_score = confluence.get('agreement_score', 0.0)
            supporting_tfs = len(confluence.get('supporting_timeframes', []))
            
            # Reward for acting when multiple timeframes agree
            if agreement_score > 0.7 and supporting_tfs >= 4:
                if action != 1:  # Taking position when confluence is strong
                    return agreement_score * 2.0
            
            # Penalty for taking position when timeframes conflict
            elif agreement_score < 0.3 and action != 1:
                return -1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Confluence reward calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_drawdown_penalty(self) -> float:
        """Calculate drawdown penalty"""
        try:
            current_portfolio = self.portfolio_value
            
            # Update max portfolio value
            if current_portfolio > self.max_portfolio_value:
                self.max_portfolio_value = current_portfolio
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.max_portfolio_value - current_portfolio) / self.max_portfolio_value
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Apply penalty for drawdown
            if self.current_drawdown > 0.05:  # More than 5% drawdown
                return -self.current_drawdown * 10.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Drawdown penalty calculation failed: {str(e)}")
            return 0.0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        try:
            if self.pytorch_available:
                # Calculate priority (TD error + small constant)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor([state]).to(self.device)
                    next_state_tensor = torch.FloatTensor([next_state]).to(self.device)
                    
                    current_q = self.q_network(state_tensor)[0][action]
                    next_q = self.target_network(next_state_tensor).max(1)[0]
                    
                    td_error = abs(reward + self.gamma * next_q * (1 - done) - current_q).item()
                    priority = td_error + 1e-6  # Small constant to ensure non-zero priority
                
                self.memory.add(state, action, reward, next_state, done, priority)
            else:
                # Fallback storage
                if not hasattr(self, 'fallback_memory'):
                    self.fallback_memory = deque(maxlen=self.memory_size)
                self.fallback_memory.append((state, action, reward, next_state, done))
                
        except Exception as e:
            self.logger.error(f"Memory storage failed: {str(e)}")
    
    def replay(self):
        """Train the network using prioritized experience replay"""
        try:
            if not self.pytorch_available:
                return self._fallback_replay()
            
            # Sample from prioritized buffer
            batch = self.memory.sample(self.batch_size, beta=0.4)
            if batch is None:
                return
            
            states, actions, rewards, next_states, dones, indices, weights = batch
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            dones_tensor = torch.BoolTensor(dones).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            
            # Current Q values
            current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
            
            # Compute loss with importance sampling weights
            td_errors = target_q_values.unsqueeze(1) - current_q_values
            loss = (td_errors.pow(2) * weights_tensor.unsqueeze(1)).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update priorities in buffer
            new_priorities = abs(td_errors.detach().cpu().numpy().flatten()) + 1e-6
            self.memory.update_priorities(indices, new_priorities)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Replay training failed: {str(e)}")
            return 0.0
    
    def _fallback_replay(self):
        """Fallback training when PyTorch not available"""
        try:
            if not hasattr(self, 'fallback_memory') or len(self.fallback_memory) < self.batch_size:
                return 0.0
            
            # Simple Q-learning update
            batch = random.sample(self.fallback_memory, min(self.batch_size, len(self.fallback_memory)))
            
            for state, action, reward, next_state, done in batch:
                # Simple state hashing for Q-table indexing
                state_hash = hash(tuple(state[:5])) % 100
                
                if not done:
                    next_state_hash = hash(tuple(next_state[:5])) % 100
                    target = reward + self.gamma * np.max(self.q_table[next_state_hash])
                else:
                    target = reward
                
                self.q_table[state_hash][action] += 0.01 * (target - self.q_table[state_hash][action])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Fallback replay failed: {str(e)}")
            return 0.0
    
    def train_agent(self, training_data: pd.DataFrame, episodes: int = 100) -> Dict[str, Any]:
        """Train the enhanced DQN agent"""
        try:
            self.logger.info(f"🚀 Starting Enhanced DQN training for {episodes} episodes")
            
            training_results = {
                'episodes_completed': 0,
                'total_reward': 0.0,
                'avg_reward_per_episode': 0.0,
                'final_epsilon': self.epsilon,
                'total_trades': 0,
                'win_rate': 0.0,
                'agent': None,
                'performance': {},
                'training_complete': False
            }
            
            episode_rewards = []
            episode_trades = []
            
            for episode in range(episodes):
                episode_reward = 0.0
                episode_trade_count = 0
                self.reset_trading_state()
                
                # Simulate trading episode
                for step in range(min(len(training_data) - 1, 1000)):  # Limit steps per episode
                    # Get current market state
                    current_data = training_data.iloc[step].to_dict()
                    next_data = training_data.iloc[step + 1].to_dict()
                    
                    # Create Elliott Wave analysis (simplified for training)
                    elliott_analysis = self._create_mock_elliott_analysis(current_data)
                    
                    # Encode state
                    state = self.encode_state(current_data, elliott_analysis)
                    next_state = self.encode_state(next_data, elliott_analysis)
                    
                    # Choose action
                    action = self.act(state, training=True)
                    
                    # Calculate reward
                    reward = self.calculate_enhanced_reward(action, next_data, elliott_analysis)
                    episode_reward += reward
                    
                    # Store experience
                    done = (step == min(len(training_data) - 2, 999))
                    self.remember(state, action, reward, next_state, done)
                    
                    # Train network
                    if len(self.memory.buffer) if self.pytorch_available else len(getattr(self, 'fallback_memory', [])) > self.batch_size:
                        loss = self.replay()
                
                # Update target network
                if self.pytorch_available and episode % self.target_update_freq == 0:
                    self.update_target_network()
                
                episode_rewards.append(episode_reward)
                episode_trades.append(episode_trade_count)
                
                # Log progress
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    self.logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            # Calculate final metrics
            training_results.update({
                'episodes_completed': episodes,
                'total_reward': sum(episode_rewards),
                'avg_reward_per_episode': np.mean(episode_rewards),
                'final_epsilon': self.epsilon,
                'total_trades': sum(episode_trades),
                'win_rate': len([r for r in episode_rewards if r > 0]) / len(episode_rewards) if episode_rewards else 0,
                'agent': self.q_network if self.pytorch_available else self.q_table,
                'performance': {
                    'episode_rewards': episode_rewards,
                    'max_reward': max(episode_rewards) if episode_rewards else 0,
                    'min_reward': min(episode_rewards) if episode_rewards else 0,
                    'std_reward': np.std(episode_rewards) if episode_rewards else 0
                },
                'training_complete': True
            })
            
            self.training_metrics.update({
                'total_episodes': episodes,
                'total_reward': sum(episode_rewards),
                'win_rate': training_results['win_rate'],
                'avg_profit_per_trade': training_results['avg_reward_per_episode']
            })
            
            self.logger.info(f"✅ Enhanced DQN training completed. Final avg reward: {training_results['avg_reward_per_episode']:.2f}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Enhanced DQN training failed: {str(e)}")
            training_results.update({
                'error': str(e),
                'training_complete': False
            })
            return training_results
    
    def _create_mock_elliott_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock Elliott Wave analysis for training"""
        # This is a simplified version for training
        # In production, this would use the actual Elliott Wave analyzer
        return {
            'confluence_analysis': {
                'overall_direction': 'UPTREND' if market_data.get('close', 0) > market_data.get('open', 0) else 'DOWNTREND',
                'strength': 5,
                'agreement_score': 0.7,
                'supporting_timeframes': ['M1', 'M5', 'M15']
            },
            'primary_wave_count': '5-WAVE',
            'fibonacci_levels': {
                'key_levels': [
                    {'distance_percent': 1.5, 'ratio': 0.618, 'level': market_data.get('close', 0) * 0.998}
                ]
            },
            'trading_signals': [
                {'type': 'BUY', 'strength': 6, 'confidence': 70}
            ]
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if self.pytorch_available:
                torch.save({
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'training_metrics': self.training_metrics,
                    'hyperparameters': {
                        'state_size': self.state_size,
                        'action_size': self.action_size,
                        'learning_rate': self.learning_rate,
                        'gamma': self.gamma,
                        'batch_size': self.batch_size
                    }
                }, filepath)
            else:
                # Save fallback model
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'q_table': self.q_table,
                        'epsilon': self.epsilon,
                        'training_metrics': self.training_metrics
                    }, f)
            
            self.logger.info(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if self.pytorch_available:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
            else:
                # Load fallback model
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data['q_table']
                    self.epsilon = data['epsilon']
                    self.training_metrics = data.get('training_metrics', self.training_metrics)
            
            self.logger.info(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

# Export class
__all__ = ['EnhancedMultiTimeframeDQNAgent']
