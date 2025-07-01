#!/usr/bin/env python3
"""
🤖 DQN REINFORCEMENT LEARNING AGENT
ตัวแทน AI สำหรับการเรียนรู้การตัดสินใจในการเทรด

Enterprise Features:
- Deep Q-Network (DQN) Implementation
- Experience Replay Buffer
- Target Network Updates
- Epsilon-Greedy Exploration
- Trading Action Selection
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
from collections import deque

# RL Imports
try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Mock PyTorch classes for type hints
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class Dropout:
            pass
    class F:
        @staticmethod
        def relu(x):
            return x

# Define DQN Network only if PyTorch is available
if PYTORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network สำหรับการเรียนรู้การเทรด"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            super(DQNNetwork, self).__init__()
            
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, action_size)
            
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return x
else:
    # Fallback DQN Network
    class DQNNetwork:
        """Fallback DQN Network เมื่อ PyTorch ไม่พร้อมใช้งาน"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_size = hidden_size
            
            # Simple neural network with numpy
            self.weights1 = np.random.normal(0, 0.1, (state_size, hidden_size))
            self.weights2 = np.random.normal(0, 0.1, (hidden_size, action_size))
            self.bias1 = np.zeros(hidden_size)  
            self.bias2 = np.zeros(action_size)
        
        def forward(self, x):
            # Simple feedforward
            if isinstance(x, np.ndarray):
                h1 = np.maximum(0, np.dot(x, self.weights1) + self.bias1)  # ReLU
                output = np.dot(h1, self.weights2) + self.bias2
                return output
            return np.zeros(self.action_size)
        
        def __call__(self, x):
            return self.forward(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience Replay Buffer สำหรับ DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """สภาพแวดล้อมการเทรดสำหรับ DQN Agent"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = 3
        
        # State space: features from data
        self.state_space = len(data.columns) - 2  # Exclude timestamp and target
    
    def reset(self):
        self.current_step = 0  
        self.balance = self.initial_balance
        self.position = 0  # 0=neutral, 1=long, -1=short
        self.position_size = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.data):
            return np.zeros(self.state_space)
        
        row = self.data.iloc[self.current_step] 
        features = [val for col, val in row.items() if col not in ['timestamp', 'target']]
        
        # Add portfolio state
        portfolio_state = [
            self.balance / self.initial_balance,
            self.position,
            self.position_size / self.initial_balance if self.position_size > 0 else 0
        ]
        
        return np.array(features + portfolio_state)
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        reward = self._execute_action(action, current_price, next_price)
        
        self.current_step += 1
        next_state = self._get_state()
        
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_reward': self.total_reward
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action, current_price, next_price):
        reward = 0
        
        if action == 1:  # Buy
            if self.position <= 0:  # Close short or open long
                if self.position < 0:  # Close short position
                    profit = (self.entry_price - current_price) * abs(self.position_size)
                    self.balance += profit
                    reward += profit / self.initial_balance
                
                # Open long position
                self.position = 1
                self.position_size = self.balance * 0.1  # 10% of balance
                self.entry_price = current_price
                
        elif action == 2:  # Sell
            if self.position >= 0:  # Close long or open short
                if self.position > 0:  # Close long position
                    profit = (current_price - self.entry_price) * self.position_size
                    self.balance += profit
                    reward += profit / self.initial_balance
                
                # Open short position
                self.position = -1
                self.position_size = self.balance * 0.1  # 10% of balance
                self.entry_price = current_price
        
        # Calculate unrealized PnL for current position
        if self.position != 0:
            unrealized_pnl = 0
            if self.position > 0:  # Long position
                unrealized_pnl = (next_price - self.entry_price) * self.position_size
            else:  # Short position
                unrealized_pnl = (self.entry_price - next_price) * abs(self.position_size)
            
            reward += unrealized_pnl / self.initial_balance * 0.1  # Small weight for unrealized
        
        self.total_reward += reward
        return reward

class DQNReinforcementAgent:
    """DQN Reinforcement Learning Agent สำหรับการเทรด"""
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # DQN Parameters
        self.state_size = None
        self.action_size = 3  # Hold, Buy, Sell
        self.memory = ReplayBuffer(10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.target_update = 100
        
        # Networks
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Training state
        self.is_trained = False
        self.training_step = 0
        
        # Check PyTorch availability
        if not PYTORCH_AVAILABLE:
            self.logger.warning("⚠️ PyTorch not available. Using fallback implementation.")
    
    def _initialize_networks(self, state_size: int):
        """เริ่มต้น Neural Networks"""
        try:
            if PYTORCH_AVAILABLE:
                self.q_network = DQNNetwork(state_size, self.action_size)
                self.target_network = DQNNetwork(state_size, self.action_size)
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                
                # Copy weights to target network
                self.target_network.load_state_dict(self.q_network.state_dict())
                
                self.logger.info("✅ DQN Networks initialized")
            else:
                # Fallback to simple rule-based agent
                self.logger.info("🔄 Using fallback rule-based agent")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize networks: {str(e)}")
            raise
    
    def train_agent(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ฝึก DQN Agent"""
        try:
            self.logger.info("🚀 Training DQN Reinforcement Learning Agent...")
            
            # Create training environment
            env = TradingEnvironment(data)
            self.state_size = env.state_space + 3  # +3 for portfolio state
            
            # Initialize networks
            self._initialize_networks(self.state_size)
            
            # Training parameters
            episodes = self.config.get('dqn', {}).get('episodes', 1000)
            max_steps = len(data) - 1
            
            # Training loop
            training_results = self._training_loop(env, episodes, max_steps)
            
            # Evaluate agent
            evaluation_results = self._evaluate_agent(env)
            
            results = {
                'success': True,
                'agent_type': 'DQN' if PYTORCH_AVAILABLE else 'Rule-based',
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'network_architecture': self._get_network_info(),
                'episodes_trained': episodes
            }
            
            self.is_trained = True
            self.logger.info("✅ DQN Agent training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ DQN training failed: {str(e)}")
            return self._train_fallback_agent(data)
    
    def _training_loop(self, env: TradingEnvironment, episodes: int, max_steps: int) -> Dict[str, Any]:
        """Training Loop หลัก"""
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_loss = 0
            
            for step in range(max_steps):
                # Select action
                action = self._select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.memory.push(state, action, reward, next_state, done)
                
                # Train if enough experiences
                if len(self.memory) > self.batch_size:
                    loss = self._replay()
                    episode_loss += loss
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_losses.append(episode_loss)
            
            # Update target network
            if episode % self.target_update == 0:
                self._update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.4f}")
        
        return {
            'final_avg_reward': float(np.mean(episode_rewards[-100:])),
            'max_reward': float(np.max(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'final_epsilon': float(self.epsilon),
            'avg_loss': float(np.mean(episode_losses)) if episode_losses else 0.0
        }
    
    def _select_action(self, state: np.ndarray) -> int:
        """เลือก Action ด้วย Epsilon-Greedy Policy"""
        if not PYTORCH_AVAILABLE:
            return self._rule_based_action(state)
        
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def _rule_based_action(self, state: np.ndarray) -> int:
        """Rule-based Action Selection (Fallback)"""
        # Simple momentum-based strategy
        if len(state) > 5:
            # Use price change as signal
            recent_change = state[4] if len(state) > 4 else 0  # Assume price change feature
            
            if recent_change > 0.001:  # 0.1% threshold
                return 1  # Buy
            elif recent_change < -0.001:
                return 2  # Sell
            else:
                return 0  # Hold
        
        return 0  # Default to hold
    
    def _replay(self) -> float:
        """Experience Replay Training"""
        if not PYTORCH_AVAILABLE:
            return 0.0
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return float(loss)
    
    def _update_target_network(self):
        """อัปเดต Target Network"""
        if PYTORCH_AVAILABLE and self.target_network is not None:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _evaluate_agent(self, env: TradingEnvironment) -> Dict[str, Any]:
        """ประเมิน Agent"""
        try:
            # Set epsilon to 0 for evaluation (no exploration)
            old_epsilon = self.epsilon
            self.epsilon = 0
            
            state = env.reset()
            total_reward = 0
            actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}
            
            while True:
                action = self._select_action(state)
                actions_taken[['hold', 'buy', 'sell'][action]] += 1
                
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Restore epsilon
            self.epsilon = old_epsilon
            
            return {
                'total_reward': float(total_reward),
                'final_balance': float(env.balance),
                'return_pct': float((env.balance - env.initial_balance) / env.initial_balance * 100),
                'actions_taken': actions_taken,
                'total_trades': sum(actions_taken.values()) - actions_taken['hold']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Agent evaluation failed: {str(e)}")
            return {'total_reward': 0, 'final_balance': 10000, 'return_pct': 0}
    
    def _train_fallback_agent(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ฝึก Fallback Agent"""
        try:
            self.logger.info("🔄 Training fallback rule-based agent...")
            
            # Simple rule-based evaluation
            env = TradingEnvironment(data)
            state = env.reset()
            total_reward = 0
            
            while True:
                action = self._rule_based_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            results = {
                'success': True,
                'agent_type': 'Rule-based (Fallback)',
                'training_results': {
                    'final_avg_reward': total_reward,
                    'max_reward': total_reward,
                    'final_epsilon': 0.0
                },
                'evaluation_results': {
                    'total_reward': total_reward,
                    'final_balance': env.balance,
                    'return_pct': (env.balance - env.initial_balance) / env.initial_balance * 100
                },
                'network_architecture': 'Rule-based momentum strategy',
                'episodes_trained': 1
            }
            
            self.is_trained = True
            self.logger.info("✅ Fallback agent training completed")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Fallback agent training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_network_info(self) -> str:
        """ข้อมูล Network Architecture"""
        if PYTORCH_AVAILABLE and self.q_network is not None:
            total_params = sum(p.numel() for p in self.q_network.parameters())
            return f"DQN with {total_params} parameters"
        else:
            return "Rule-based agent (no neural network)"
    
    def predict_action(self, state: np.ndarray) -> int:
        """ทำนาย Action สำหรับ State ที่ให้"""
        if not self.is_trained:
            return 0  # Default to hold
        
        return self._select_action(state)
    
    def save_agent(self, filepath: str):
        """บันทึก Agent"""
        try:
            import joblib
            
            agent_data = {
                'config': self.config,
                'is_trained': self.is_trained,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'pytorch_available': PYTORCH_AVAILABLE
            }
            
            if PYTORCH_AVAILABLE and self.q_network is not None:
                agent_data['q_network_state'] = self.q_network.state_dict()
                agent_data['target_network_state'] = self.target_network.state_dict()
            
            joblib.dump(agent_data, filepath)
            self.logger.info(f"💾 Agent saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save agent: {str(e)}")
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """สรุปข้อมูล Agent"""
        return {
            'agent_type': 'DQN Reinforcement Learning Agent',
            'is_trained': self.is_trained,
            'pytorch_available': PYTORCH_AVAILABLE,
            'action_space': self.action_size,
            'state_space': self.state_size,
            'current_epsilon': self.epsilon,
            'training_steps': self.training_step,
            'features': [
                'Deep Q-Network (DQN)',
                'Experience Replay Buffer',
                'Target Network Updates',
                'Epsilon-Greedy Exploration',
                'Trading Action Selection'
            ]
        }
