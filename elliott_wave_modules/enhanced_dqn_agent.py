#!/usr/bin/env python3
"""
🎯 ENHANCED DQN REINFORCEMENT LEARNING AGENT
DQN Agent ขั้นสูงสำหรับการเทรด Elliott Wave พร้อม Curriculum Learning

Enhanced Features:
- Elliott Wave-based Reward System
- Curriculum Learning (ง่าย → ยาก)
- Multi-timeframe State Representation
- Advanced Action Space (Buy/Sell/Hold + Position Sizing)
- Risk-adjusted Reward Calculation
- Dynamic Exploration Strategy
"""

import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import random
from collections import deque
from enum import Enum
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Advanced Logging Integration
try:
    from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

# Import Elliott Wave Analyzer
try:
    from elliott_wave_modules.advanced_elliott_wave_analyzer import (
        AdvancedElliottWaveAnalyzer, MultiTimeframeWaveAnalysis, 
        WaveDirection, WaveType, WavePosition
    )
    ELLIOTT_WAVE_AVAILABLE = True
except ImportError:
    ELLIOTT_WAVE_AVAILABLE = False

# PyTorch availability check
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    pass


class ActionType(Enum):
    """ประเภทของ Action"""
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


class DifficultyLevel(Enum):
    """ระดับความยากของ Curriculum Learning"""
    BEGINNER = "beginner"      # เทรนด์ชัด, สัญญาณง่าย
    INTERMEDIATE = "intermediate"  # mixed market conditions
    ADVANCED = "advanced"      # ช่วงขายข้าง, สัญญาณยาก
    EXPERT = "expert"          # สภาวะตลาดซับซ้อน


class CurriculumLearningManager:
    """จัดการ Curriculum Learning สำหรับ DQN"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.current_level = DifficultyLevel.BEGINNER
        self.performance_threshold = 0.75  # 75% win rate to advance
        self.episodes_per_level = self.config.get('episodes_per_level', 50)
        self.current_episode_in_level = 0
        self.level_performance_history = []
        
    def should_advance_level(self, recent_performance: float) -> bool:
        """ตรวจสอบว่าควรเลื่อนระดับหรือไม่"""
        self.level_performance_history.append(recent_performance)
        
        # Need minimum episodes and good performance
        if (self.current_episode_in_level >= self.episodes_per_level and
            len(self.level_performance_history) >= 10):
            
            recent_avg = np.mean(self.level_performance_history[-10:])
            return recent_avg >= self.performance_threshold
        
        return False
    
    def advance_to_next_level(self):
        """เลื่อนไประดับถัดไป"""
        level_order = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, 
                      DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        
        current_index = level_order.index(self.current_level)
        if current_index < len(level_order) - 1:
            self.current_level = level_order[current_index + 1]
            self.current_episode_in_level = 0
            self.level_performance_history = []
            return True
        return False
    
    def get_training_data_for_level(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """เลือกข้อมูลฝึกตามระดับความยาก"""
        if self.current_level == DifficultyLevel.BEGINNER:
            # เลือกช่วงที่เทรนด์ชัด
            return self._select_trending_periods(full_data, min_trend_strength=0.7)
        
        elif self.current_level == DifficultyLevel.INTERMEDIATE:
            # เลือกข้อมูลแบบผสม
            trending = self._select_trending_periods(full_data, min_trend_strength=0.5)
            sideways = self._select_sideways_periods(full_data, max_trend_strength=0.3)
            return pd.concat([trending[:len(trending)//2], sideways[:len(sideways)//2]])
        
        elif self.current_level == DifficultyLevel.ADVANCED:
            # เลือกช่วงขายข้างเป็นหลัก
            return self._select_sideways_periods(full_data, max_trend_strength=0.4)
        
        else:  # EXPERT
            # ใช้ข้อมูลทั้งหมด รวมถึงช่วงที่ซับซ้อน
            return full_data
    
    def _select_trending_periods(self, data: pd.DataFrame, min_trend_strength: float) -> pd.DataFrame:
        """เลือกช่วงที่มีเทรนด์ชัด"""
        # Calculate trend strength using moving averages
        data['ema_fast'] = data['close'].ewm(span=12).mean()
        data['ema_slow'] = data['close'].ewm(span=26).mean()
        data['trend_strength'] = abs(data['ema_fast'] - data['ema_slow']) / data['close']
        
        # Select periods with strong trend
        trending_mask = data['trend_strength'] >= min_trend_strength
        return data[trending_mask].copy()
    
    def _select_sideways_periods(self, data: pd.DataFrame, max_trend_strength: float) -> pd.DataFrame:
        """เลือกช่วงขายข้าง"""
        data['ema_fast'] = data['close'].ewm(span=12).mean()
        data['ema_slow'] = data['close'].ewm(span=26).mean()
        data['trend_strength'] = abs(data['ema_fast'] - data['ema_slow']) / data['close']
        
        # Select periods with weak trend (sideways)
        sideways_mask = data['trend_strength'] <= max_trend_strength
        return data[sideways_mask].copy()


if PYTORCH_AVAILABLE:
    class EnhancedDQNNetwork(nn.Module):
        """Enhanced DQN Network ที่รองรับ Multi-timeframe Input"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
            super(EnhancedDQNNetwork, self).__init__()
            
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = hidden_size
            
            # Larger network for complex Elliott Wave patterns
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
            self.fc5 = nn.Linear(hidden_size // 4, action_size)
            
            self.dropout = nn.Dropout(0.3)
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)
            self.batch_norm2 = nn.BatchNorm1d(hidden_size)
            self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
        
        def forward(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            x = F.relu(self.batch_norm1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.batch_norm2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.batch_norm3(self.fc3(x)))
            x = self.dropout(x)
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x
        
        def _initialize_numpy_network(self):
            """Initialize numpy-based network"""
            return {
                'w1': np.random.normal(0, 0.1, (self.state_size, self.hidden_size)),
                'b1': np.zeros(self.hidden_size),
                'w2': np.random.normal(0, 0.1, (self.hidden_size, self.action_size)),
                'b2': np.zeros(self.action_size)
            }
        
        def _numpy_forward(self, x):
            """Numpy forward pass"""
            if isinstance(x, np.ndarray):
                h1 = np.maximum(0, np.dot(x, self.weights['w1']) + self.weights['b1'])
                output = np.dot(h1, self.weights['w2']) + self.weights['b2']
                return output
            return np.zeros(self.action_size)

else:
    # Fallback class when PyTorch is not available
    class EnhancedDQNNetwork:
        """Fallback Enhanced DQN Network using numpy"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = hidden_size
            self.weights = self._initialize_numpy_network()
        
        def _initialize_numpy_network(self):
            """Initialize numpy-based network"""
            return {
                'w1': np.random.normal(0, 0.1, (self.state_size, self.hidden_size)),
                'b1': np.zeros(self.hidden_size),
                'w2': np.random.normal(0, 0.1, (self.hidden_size, self.action_size)),
                'b2': np.zeros(self.action_size)
            }
        
        def __call__(self, x):
            """Forward pass using numpy"""
            if isinstance(x, np.ndarray):
                h1 = np.maximum(0, np.dot(x, self.weights['w1']) + self.weights['b1'])
                output = np.dot(h1, self.weights['w2']) + self.weights['b2']
                return output
            return np.zeros(self.action_size)


class ElliottWaveRewardCalculator:
    """คำนวณ Reward จาก Elliott Wave Analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.base_reward_scale = self.config.get('base_reward_scale', 100.0)
        self.wave_alignment_weight = self.config.get('wave_alignment_weight', 0.3)
        self.fibonacci_confluence_weight = self.config.get('fibonacci_confluence_weight', 0.2)
        self.trend_strength_weight = self.config.get('trend_strength_weight', 0.2)
        self.risk_penalty_weight = self.config.get('risk_penalty_weight', 0.3)
        
    def calculate_reward(self, action: ActionType, price_change: float, 
                        wave_analysis: Optional[MultiTimeframeWaveAnalysis] = None,
                        position_size: float = 0.1) -> float:
        """คำนวณ reward จาก action และ Elliott Wave analysis"""
        
        # Base reward from price movement
        base_reward = self._calculate_base_reward(action, price_change, position_size)
        
        # Elliott Wave enhancement
        if wave_analysis and ELLIOTT_WAVE_AVAILABLE:
            wave_reward = self._calculate_wave_reward(action, wave_analysis)
            base_reward += wave_reward
        
        # Risk adjustment
        risk_penalty = self._calculate_risk_penalty(action, price_change, position_size)
        
        total_reward = base_reward - risk_penalty
        
        # Clamp reward to reasonable range
        return np.clip(total_reward, -1000.0, 1000.0)
    
    def _calculate_base_reward(self, action: ActionType, price_change: float, position_size: float) -> float:
        """คำนวณ base reward จากการเคลื่อนไหวของราคา"""
        reward = 0.0
        
        # Buy actions
        if action in [ActionType.BUY_SMALL, ActionType.BUY_MEDIUM, ActionType.BUY_LARGE]:
            reward = price_change * self.base_reward_scale * position_size
            
        # Sell actions  
        elif action in [ActionType.SELL_SMALL, ActionType.SELL_MEDIUM, ActionType.SELL_LARGE]:
            reward = -price_change * self.base_reward_scale * position_size
            
        # Hold action
        else:  # HOLD
            # Small penalty for inaction, but reward if market is stable
            if abs(price_change) < 0.001:  # Very stable
                reward = 5.0  # Reward for holding during stability
            else:
                reward = -1.0  # Small penalty for missing opportunities
        
        return reward
    
    def _calculate_wave_reward(self, action: ActionType, wave_analysis: MultiTimeframeWaveAnalysis) -> float:
        """คำนวณ reward จาก Elliott Wave analysis"""
        wave_reward = 0.0
        
        # Alignment bonus
        alignment_bonus = wave_analysis.wave_alignment_score * self.wave_alignment_weight * 50.0
        
        # Fibonacci confluence bonus
        confluence_bonus = len(wave_analysis.fibonacci_confluence) * self.fibonacci_confluence_weight * 20.0
        
        # Trend strength bonus
        trend_bonus = wave_analysis.trend_strength * self.trend_strength_weight * 30.0
        
        # Action-specific wave rewards
        primary_wave = wave_analysis.primary_wave
        
        # Reward actions that align with wave direction
        if action in [ActionType.BUY_SMALL, ActionType.BUY_MEDIUM, ActionType.BUY_LARGE]:
            if primary_wave.direction == WaveDirection.UP:
                wave_reward += alignment_bonus + confluence_bonus + trend_bonus
                
                # Extra bonus for buying at wave 1 or 3 (strongest impulse waves)
                if primary_wave.wave_position in [WavePosition.WAVE_1, WavePosition.WAVE_3]:
                    wave_reward += 25.0
                    
            elif primary_wave.direction == WaveDirection.DOWN:
                wave_reward -= (alignment_bonus + confluence_bonus) * 0.5  # Penalty for wrong direction
        
        elif action in [ActionType.SELL_SMALL, ActionType.SELL_MEDIUM, ActionType.SELL_LARGE]:
            if primary_wave.direction == WaveDirection.DOWN:
                wave_reward += alignment_bonus + confluence_bonus + trend_bonus
                
                # Extra bonus for selling at wave A or C (corrective waves)
                if primary_wave.wave_position in [WavePosition.WAVE_A, WavePosition.WAVE_C]:
                    wave_reward += 25.0
                    
            elif primary_wave.direction == WaveDirection.UP:
                wave_reward -= (alignment_bonus + confluence_bonus) * 0.5
        
        else:  # HOLD
            # Reward holding during corrective wave B or wave 4
            if primary_wave.wave_position in [WavePosition.WAVE_B, WavePosition.WAVE_4]:
                wave_reward += 15.0
            # Reward holding during sideways movement
            elif primary_wave.direction == WaveDirection.SIDEWAYS:
                wave_reward += 10.0
        
        return wave_reward
    
    def _calculate_risk_penalty(self, action: ActionType, price_change: float, position_size: float) -> float:
        """คำนวณ risk penalty"""
        penalty = 0.0
        
        # Position size penalty for aggressive actions without justification
        if action in [ActionType.BUY_LARGE, ActionType.SELL_LARGE]:
            penalty += 10.0 * position_size  # Higher penalty for large positions
        
        # Volatility penalty
        if abs(price_change) > 0.05:  # High volatility (5%+)
            penalty += abs(price_change) * 100 * position_size
        
        return penalty * self.risk_penalty_weight


class EnhancedDQNAgent:
    """Enhanced DQN Agent พร้อม Elliott Wave Integration และ Curriculum Learning"""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        
        # Initialize Advanced Logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_unified_logger()
            self.logger.info("🚀 Enhanced DQN Agent initialized", component="Enhanced_DQN")
        else:
            self.logger = logger or get_unified_logger()
        
        # Enhanced DQN Parameters
        self.state_size = self.config.get('state_size', 50)  # Larger state for Elliott Wave features
        self.action_size = len(ActionType)  # 7 actions
        self.learning_rate = self.config.get('learning_rate', 0.0005)  # Lower for stability
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.memory_size = self.config.get('memory_size', 20000)  # Larger memory
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_frequency = self.config.get('target_update_frequency', 100)
        
        # Elliott Wave Integration
        if ELLIOTT_WAVE_AVAILABLE:
            self.elliott_wave_analyzer = AdvancedElliottWaveAnalyzer(config=self.config)
            self.logger.info("✅ Elliott Wave Analyzer integrated")
        else:
            self.elliott_wave_analyzer = None
            self.logger.warning("⚠️ Elliott Wave Analyzer not available")
        
        # Curriculum Learning
        self.curriculum_manager = CurriculumLearningManager(config=self.config)
        self.logger.info(f"📚 Curriculum Learning initialized at {self.curriculum_manager.current_level.value} level")
        
        # Reward Calculator
        self.reward_calculator = ElliottWaveRewardCalculator(config=self.config)
        
        # Trading state
        self.current_position = 0.0  # -1 to 1 (short to long)
        self.position_history = []
        self.balance = self.config.get('initial_balance', 10000.0)
        self.initial_balance = self.balance
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_wins = []
        self.training_step = 0
    
    def _initialize_components(self):
        """เริ่มต้นส่วนประกอบต่างๆ"""
        try:
            # Initialize networks
            self.q_network = EnhancedDQNNetwork(self.state_size, self.action_size)
            self.target_network = EnhancedDQNNetwork(self.state_size, self.action_size)
            
            # Initialize replay buffer
            self.replay_buffer = deque(maxlen=self.memory_size)
            
            if PYTORCH_AVAILABLE:
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.is_trained = False
            self.episode_count = 0
            
            self.logger.info(f"✅ Enhanced DQN components initialized (PyTorch: {PYTORCH_AVAILABLE})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced DQN: {str(e)}")
            raise
    
    def get_enhanced_state(self, market_data: pd.DataFrame, current_idx: int) -> np.ndarray:
        """สร้าง enhanced state ที่รวม Elliott Wave features"""
        try:
            # Get basic market features
            basic_features = self._get_basic_market_features(market_data, current_idx)
            
            # Get Elliott Wave features if available
            elliott_features = np.zeros(20)  # Default Elliott Wave features
            if self.elliott_wave_analyzer and len(market_data) > current_idx + 100:
                try:
                    # Use last 200 bars for Elliott Wave analysis
                    analysis_data = market_data.iloc[max(0, current_idx-199):current_idx+1]
                    wave_analysis = self.elliott_wave_analyzer.analyze_multi_timeframe_waves(analysis_data)
                    
                    # Convert wave analysis to features
                    wave_features = self.elliott_wave_analyzer.generate_elliott_wave_features(wave_analysis)
                    elliott_features = self._convert_wave_features_to_array(wave_features)
                    
                except Exception as e:
                    self.logger.debug(f"Elliott Wave analysis failed: {str(e)}")
            
            # Get position and portfolio features
            portfolio_features = self._get_portfolio_features()
            
            # Combine all features
            state = np.concatenate([basic_features, elliott_features, portfolio_features])
            
            # Ensure state size matches expected size
            if len(state) != self.state_size:
                # Pad or trim to match expected size
                if len(state) < self.state_size:
                    state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                else:
                    state = state[:self.state_size]
            
            # Normalize state
            state = self._normalize_state(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ State preparation failed: {str(e)}")
            return np.zeros(self.state_size)
    
    def _get_basic_market_features(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """ได้ basic market features"""
        try:
            # Get last 20 bars of price data
            lookback = min(20, idx + 1)
            recent_data = data.iloc[max(0, idx-lookback+1):idx+1]
            
            features = []
            
            if len(recent_data) > 0:
                # Price features
                close_prices = recent_data['close'].values
                features.extend(close_prices[-10:] if len(close_prices) >= 10 else 
                              np.pad(close_prices, (10-len(close_prices), 0), 'edge'))
                
                # Technical indicators
                if len(close_prices) >= 5:
                    sma_5 = np.mean(close_prices[-5:])
                    sma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else sma_5
                    rsi = self._calculate_simple_rsi(close_prices)
                    
                    features.extend([sma_5, sma_10, rsi])
                else:
                    features.extend([0.0, 0.0, 50.0])  # Default values
                
                # Price changes
                if len(close_prices) >= 2:
                    price_change_1 = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
                    features.append(price_change_1)
                else:
                    features.append(0.0)
            else:
                features = [0.0] * 14  # Default feature size
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.debug(f"Basic features calculation failed: {str(e)}")
            return np.zeros(14, dtype=np.float32)
    
    def _convert_wave_features_to_array(self, wave_features: Dict[str, float]) -> np.ndarray:
        """แปลง Elliott Wave features เป็น array"""
        # Key features to extract (total 20 features)
        key_features = [
            'primary_wave_direction', 'primary_wave_confidence', 'primary_wave_fibonacci_ratio',
            'primary_wave_price_change_pct', 'primary_wave_volume_confirmation',
            'primary_wave_is_impulse', 'primary_wave_is_corrective',
            'primary_wave_is_wave_1', 'primary_wave_is_wave_2', 'primary_wave_is_wave_3',
            'primary_wave_is_wave_4', 'primary_wave_is_wave_5', 'primary_wave_is_wave_a',
            'primary_wave_is_wave_b', 'primary_wave_is_wave_c',
            'wave_alignment_score', 'fibonacci_confluence_count', 'trend_strength',
            'overall_confidence', 'recommended_action_numeric'
        ]
        
        features = []
        for key in key_features:
            features.append(wave_features.get(key, 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """ได้ portfolio และ position features"""
        features = [
            self.current_position,  # Current position (-1 to 1)
            self.balance / self.initial_balance - 1.0,  # Portfolio return
            len(self.position_history) / 1000.0,  # Number of trades (normalized)
        ]
        return np.array(features, dtype=np.float32)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state features"""
        try:
            # Handle NaN and infinity
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Simple normalization: clip to reasonable range
            state = np.clip(state, -10.0, 10.0)
            
            return state
            
        except Exception:
            return np.zeros_like(state)
    
    def _calculate_simple_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """คำนวณ RSI แบบง่าย"""
        try:
            if len(prices) < 2:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
            avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return 50.0
    
    def get_action(self, state: np.ndarray, training: bool = True) -> ActionType:
        """เลือก action ด้วย epsilon-greedy policy"""
        try:
            # Epsilon-greedy exploration (only during training)
            if training and np.random.random() < self.epsilon:
                return ActionType(np.random.randint(0, self.action_size))
            
            # Get Q-values
            if PYTORCH_AVAILABLE:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
            else:
                q_values = self.q_network.forward(state)
                action_idx = np.argmax(q_values)
            
            return ActionType(action_idx)
            
        except Exception as e:
            self.logger.error(f"❌ Action selection failed: {str(e)}")
            return ActionType.HOLD
    
    def train_with_curriculum(self, full_training_data: pd.DataFrame, 
                            total_episodes: int = 1000) -> Dict[str, Any]:
        """ฝึก DQN Agent ด้วย Curriculum Learning"""
        try:
            self.logger.info(f"🚀 Starting Enhanced DQN training with Curriculum Learning...")
            self.logger.info(f"📚 Total episodes: {total_episodes}")
            
            training_results = {
                'episodes_completed': 0,
                'curriculum_levels_completed': [],
                'final_performance': 0.0,
                'episode_rewards': [],
                'level_progressions': []
            }
            
            episodes_completed = 0
            
            while episodes_completed < total_episodes:
                # Get training data for current curriculum level
                level_data = self.curriculum_manager.get_training_data_for_level(full_training_data)
                
                self.logger.info(f"📚 Training at {self.curriculum_manager.current_level.value} level")
                self.logger.info(f"📊 Level data size: {len(level_data)} bars")
                
                # Train episodes at current level
                level_episodes = min(self.curriculum_manager.episodes_per_level, 
                                   total_episodes - episodes_completed)
                
                level_rewards = []
                level_wins = []
                
                for episode in range(level_episodes):
                    episode_result = self.train_episode(level_data)
                    level_rewards.append(episode_result.get('reward', 0.0))
                    level_wins.append(1.0 if episode_result.get('reward', 0.0) > 0 else 0.0)
                    
                    episodes_completed += 1
                    self.curriculum_manager.current_episode_in_level += 1
                    
                    # Log progress
                    if episode % 10 == 0:
                        recent_performance = np.mean(level_wins[-10:]) if level_wins else 0.0
                        self.logger.info(f"📈 Episode {episodes_completed}: Level performance: {recent_performance:.2f}")
                
                # Check if should advance curriculum level
                level_performance = np.mean(level_wins) if level_wins else 0.0
                
                if self.curriculum_manager.should_advance_level(level_performance):
                    if self.curriculum_manager.advance_to_next_level():
                        self.logger.info(f"🎓 Advanced to {self.curriculum_manager.current_level.value} level!")
                        training_results['level_progressions'].append({
                            'episode': episodes_completed,
                            'from_level': self.curriculum_manager.current_level.value,
                            'performance': level_performance
                        })
                
                training_results['episode_rewards'].extend(level_rewards)
            
            # Final evaluation
            self.is_trained = True
            final_performance = np.mean(level_wins[-50:]) if len(level_wins) >= 50 else np.mean(level_wins)
            
            training_results.update({
                'episodes_completed': episodes_completed,
                'final_performance': final_performance,
                'final_curriculum_level': self.curriculum_manager.current_level.value,
                'success': True
            })
            
            self.logger.info(f"✅ Enhanced DQN training completed!")
            self.logger.info(f"📊 Final performance: {final_performance:.3f}")
            self.logger.info(f"🎓 Final curriculum level: {self.curriculum_manager.current_level.value}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced DQN training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_episode(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ฝึกหนึ่ง episode"""
        try:
            episode_reward = 0.0
            episode_length = min(500, len(data) - 50)  # Leave room for Elliott Wave analysis
            
            # Reset episode state
            self.current_position = 0.0
            episode_balance = self.balance
            
            step = 0
            for step in range(episode_length):
                if step + 50 >= len(data):  # Ensure enough data for analysis
                    break
                
                # Get current state
                state = self.get_enhanced_state(data, step + 50)
                
                # Get action
                action = self.get_action(state, training=True)
                
                # Execute action and calculate reward
                next_state, reward, done = self._execute_action_with_elliott_wave(
                    data, step + 50, action, state
                )
                
                # Store experience
                self.replay_buffer.append((state, action.value, reward, next_state, done))
                
                # Train if enough experiences
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step()
                
                episode_reward += reward
                
                if done:
                    break
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network
            if PYTORCH_AVAILABLE and self.training_step % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.episode_count += 1
            
            return {
                'episode': self.episode_count,
                'reward': episode_reward,
                'epsilon': self.epsilon,
                'steps': step + 1,
                'balance_change': self.balance - episode_balance
            }
            
        except Exception as e:
            self.logger.error(f"❌ Episode training failed: {str(e)}")
            return {'episode': self.episode_count, 'reward': 0.0, 'error': str(e)}
    
    def _execute_action_with_elliott_wave(self, data: pd.DataFrame, current_idx: int, 
                                        action: ActionType, current_state: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action พร้อม Elliott Wave analysis สำหรับ reward calculation"""
        try:
            # Get next state
            if current_idx + 1 < len(data):
                next_state = self.get_enhanced_state(data, current_idx + 1)
                done = False
            else:
                next_state = np.zeros(self.state_size)
                done = True
            
            # Calculate price change
            if current_idx + 1 < len(data):
                current_price = data.iloc[current_idx]['close']
                next_price = data.iloc[current_idx + 1]['close']
                price_change = (next_price - current_price) / current_price
            else:
                price_change = 0.0
            
            # Get Elliott Wave analysis for reward calculation
            wave_analysis = None
            if self.elliott_wave_analyzer and current_idx >= 100:
                try:
                    analysis_data = data.iloc[current_idx-199:current_idx+1]
                    wave_analysis = self.elliott_wave_analyzer.analyze_multi_timeframe_waves(analysis_data)
                except Exception:
                    pass
            
            # Calculate position size based on action
            position_size = self._get_position_size_for_action(action)
            
            # Calculate enhanced reward
            reward = self.reward_calculator.calculate_reward(
                action, price_change, wave_analysis, position_size
            )
            
            # Update position and balance
            self._update_trading_state(action, current_price, next_price, position_size)
            
            return next_state, reward, done
            
        except Exception as e:
            self.logger.error(f"❌ Action execution failed: {str(e)}")
            return np.zeros(self.state_size), 0.0, True
    
    def _get_position_size_for_action(self, action: ActionType) -> float:
        """ได้ position size จาก action type"""
        size_map = {
            ActionType.HOLD: 0.0,
            ActionType.BUY_SMALL: 0.25,
            ActionType.BUY_MEDIUM: 0.5,
            ActionType.BUY_LARGE: 1.0,
            ActionType.SELL_SMALL: 0.25,
            ActionType.SELL_MEDIUM: 0.5,
            ActionType.SELL_LARGE: 1.0
        }
        return size_map.get(action, 0.0)
    
    def _update_trading_state(self, action: ActionType, current_price: float, 
                            next_price: float, position_size: float):
        """อัพเดท trading state"""
        try:
            price_change = (next_price - current_price) / current_price
            
            # Update position
            if action in [ActionType.BUY_SMALL, ActionType.BUY_MEDIUM, ActionType.BUY_LARGE]:
                position_change = position_size
                self.current_position = min(1.0, self.current_position + position_change)
                
            elif action in [ActionType.SELL_SMALL, ActionType.SELL_MEDIUM, ActionType.SELL_LARGE]:
                position_change = -position_size
                self.current_position = max(-1.0, self.current_position + position_change)
            
            # Update balance based on position and price change
            position_pnl = self.current_position * price_change * self.balance * 0.1  # 10% max exposure
            self.balance += position_pnl
            
            # Record trade
            if action != ActionType.HOLD:
                self.position_history.append({
                    'action': action.name,
                    'price': current_price,
                    'position_size': position_size,
                    'pnl': position_pnl
                })
        
        except Exception as e:
            self.logger.debug(f"Trading state update failed: {str(e)}")
    
    def _train_step(self):
        """ฝึกโมเดล DQN หนึ่งขั้นตอน"""
        try:
            if len(self.replay_buffer) < self.batch_size:
                return
            
            # Sample batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            
            if PYTORCH_AVAILABLE:
                self._pytorch_train_step(states, actions, rewards, next_states, dones)
            
            self.training_step += 1
            
        except Exception as e:
            self.logger.error(f"❌ Training step failed: {str(e)}")
    
    def _pytorch_train_step(self, states, actions, rewards, next_states, dones):
        """PyTorch training step"""
        try:
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.huber_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        except Exception as e:
            self.logger.error(f"❌ PyTorch training step failed: {str(e)}")
    
    def predict_action(self, market_data: pd.DataFrame, current_idx: int) -> ActionType:
        """ทำนาย action สำหรับการใช้งานจริง"""
        try:
            state = self.get_enhanced_state(market_data, current_idx)
            return self.get_action(state, training=False)
        except Exception as e:
            self.logger.error(f"❌ Action prediction failed: {str(e)}")
            return ActionType.HOLD
    
    def get_agent_info(self) -> Dict[str, Any]:
        """ข้อมูลเกี่ยวกับ Enhanced DQN Agent"""
        return {
            'agent_type': 'Enhanced DQN with Elliott Wave Integration',
            'is_trained': self.is_trained,
            'pytorch_available': PYTORCH_AVAILABLE,
            'elliott_wave_available': ELLIOTT_WAVE_AVAILABLE,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'current_curriculum_level': self.curriculum_manager.current_level.value,
            'current_position': self.current_position,
            'balance': self.balance,
            'total_trades': len(self.position_history),
            'features': [
                'Enhanced State Representation',
                'Elliott Wave Integration',
                'Curriculum Learning',
                'Advanced Action Space',
                'Risk-adjusted Rewards',
                'Multi-timeframe Analysis'
            ]
        }


# Factory function
def create_enhanced_dqn_agent(config: Dict = None) -> EnhancedDQNAgent:
    """สร้าง Enhanced DQN Agent instance"""
    default_config = {
        'state_size': 50,
        'learning_rate': 0.0005,
        'episodes_per_level': 100,
        'base_reward_scale': 100.0,
        'timeframes': ['1min', '5min', '15min', '1H']
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedDQNAgent(config=default_config)


if __name__ == "__main__":
    # Test the enhanced agent
    print("🧪 Testing Enhanced DQN Agent")
    agent = create_enhanced_dqn_agent()
    print("✅ Enhanced DQN Agent created successfully")
    print(f"📊 Agent info: {agent.get_agent_info()}")
