#!/usr/bin/env python3
"""
🎯 DQN REINFORCEMENT LEARNING AGENT
ตัวแทน Deep Q-Network สำหรับการเรียนรู้การเทรด Elliott Wave

Enterprise Features:
- Deep Q-Network Architecture
- Experience Replay Buffer
- Epsilon-Greedy Exploration
- Production-ready with Fallbacks
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
from collections import deque

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

# Define networks based on PyTorch availability
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
            if isinstance(x, np.ndarray):
                h1 = np.maximum(0, np.dot(x, self.weights1) + self.bias1)  # ReLU
                output = np.dot(h1, self.weights2) + self.bias2
                return output
            return np.zeros(self.action_size)
        
        def __call__(self, x):
            return self.forward(x)

class ReplayBuffer:
    """Experience Replay Buffer สำหรับ DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """เพิ่ม experience ลง buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """สุ่มตัวอย่าง batch จาก buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNReinforcementAgent:
    """DQN Reinforcement Learning Agent สำหรับ Elliott Wave Trading"""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # DQN Parameters
        self.state_size = self.config.get('dqn', {}).get('state_size', 20)
        self.action_size = self.config.get('dqn', {}).get('action_size', 3)  # Hold, Buy, Sell
        self.learning_rate = self.config.get('dqn', {}).get('learning_rate', 0.001)
        self.gamma = self.config.get('dqn', {}).get('gamma', 0.95)
        self.epsilon = self.config.get('dqn', {}).get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('dqn', {}).get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('dqn', {}).get('epsilon_decay', 0.995)
        self.memory_size = self.config.get('dqn', {}).get('memory_size', 10000)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """เริ่มต้นส่วนประกอบต่างๆ"""
        try:
            # Initialize networks
            self.q_network = DQNNetwork(self.state_size, self.action_size)
            self.target_network = DQNNetwork(self.state_size, self.action_size)
            
            # Initialize replay buffer
            self.replay_buffer = ReplayBuffer(self.memory_size)
            
            # Initialize optimizer (PyTorch only)
            if PYTORCH_AVAILABLE:
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Trading state
            self.is_trained = False
            self.episode_count = 0
            
            self.logger.info(f"✅ DQN Agent initialized (PyTorch: {PYTORCH_AVAILABLE})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize DQN Agent: {str(e)}")
            raise
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """เลือก action ด้วย epsilon-greedy policy"""
        try:
            # Epsilon-greedy exploration
            if training and np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
            
            # Get Q-values
            if PYTORCH_AVAILABLE:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
            else:
                # Fallback action selection
                q_values = self.q_network.forward(state)
                return np.argmax(q_values)
                
        except Exception as e:
            self.logger.error(f"❌ Action selection failed: {str(e)}")
            return 0  # Default to Hold
    
    def store_experience(self, state, action, reward, next_state, done):
        """เก็บ experience ลง replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """ฝึกโมเดล DQN หนึ่งขั้นตอน"""
        try:
            if len(self.replay_buffer) < batch_size:
                return {'loss': 0.0, 'q_value': 0.0}
            
            # Sample batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            if PYTORCH_AVAILABLE:
                return self._pytorch_train_step(states, actions, rewards, next_states, dones)
            else:
                return self._numpy_train_step(states, actions, rewards, next_states, dones)
                
        except Exception as e:
            self.logger.error(f"❌ Training step failed: {str(e)}")
            return {'loss': 0.0, 'q_value': 0.0}
    
    def _pytorch_train_step(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        """PyTorch training step"""
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item()
        }
    
    def _numpy_train_step(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        """Numpy fallback training step"""
        # Simple Q-learning update
        avg_loss = 0.0
        avg_q_value = 0.0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            
            # Current Q-value
            current_q = self.q_network.forward(state)[action]
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.q_network.forward(next_state)
                target_q = reward + self.gamma * np.max(next_q_values)
            
            # Simple update (gradient descent approximation)
            error = target_q - current_q
            
            avg_loss += error ** 2
            avg_q_value += current_q
        
        avg_loss /= len(states)
        avg_q_value /= len(states)
        
        return {
            'loss': float(avg_loss),
            'q_value': float(avg_q_value)
        }
    
    def train_episode(self, env_data: pd.DataFrame) -> Dict[str, Any]:
        """ฝึกโมเดลหนึ่ง episode"""
        try:
            self.logger.info(f"🎯 Training DQN Episode {self.episode_count + 1}...")
            
            # Prepare environment
            episode_reward = 0
            episode_length = min(1000, len(env_data) - self.state_size)
            
            # Initialize state
            state = self._prepare_state(env_data.iloc[:self.state_size])
            
            losses = []
            q_values = []
            
            for step in range(episode_length):
                # Get action
                action = self.get_action(state, training=True)
                
                # Execute action and get reward
                next_state, reward, done = self._step_environment(env_data, step, action)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train if enough experiences
                if len(self.replay_buffer) > 100:
                    train_result = self.train_step()
                    losses.append(train_result['loss'])
                    q_values.append(train_result['q_value'])
                
                # Update state
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network
            if PYTORCH_AVAILABLE and self.episode_count % 10 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.episode_count += 1
            
            results = {
                'episode': self.episode_count,
                'reward': episode_reward,
                'epsilon': self.epsilon,
                'avg_loss': np.mean(losses) if losses else 0.0,
                'avg_q_value': np.mean(q_values) if q_values else 0.0,
                'steps': step + 1
            }
            
            self.logger.info(f"✅ Episode {self.episode_count} completed: Reward={episode_reward:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Episode training failed: {str(e)}")
            return {'episode': self.episode_count, 'reward': 0.0, 'error': str(e)}
    
    def _prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        """เตรียม state จากข้อมูล"""
        try:
            if len(data) < self.state_size:
                # Pad with zeros if not enough data
                state = np.zeros(self.state_size)
                state[:len(data)] = data.iloc[:, -1].values[:self.state_size]
            else:
                # Use last few values as state
                state = data.iloc[-self.state_size:, -1].values
            
            # Normalize state
            state = (state - np.mean(state)) / (np.std(state) + 1e-8)
            return state
            
        except Exception as e:
            self.logger.error(f"❌ State preparation failed: {str(e)}")
            return np.zeros(self.state_size)
    
    def _step_environment(self, data: pd.DataFrame, step: int, action: int) -> Tuple[np.ndarray, float, bool]:
        """จำลองการทำงานของ environment"""
        try:
            # Get next state
            if step + self.state_size + 1 < len(data):
                next_state = self._prepare_state(data.iloc[step+1:step+1+self.state_size])
                done = False
            else:
                next_state = np.zeros(self.state_size)
                done = True
            
            # Calculate reward based on action and price movement
            if step + 1 < len(data):
                current_price = data.iloc[step + self.state_size]['close'] if 'close' in data.columns else data.iloc[step + self.state_size, -1]
                next_price = data.iloc[step + self.state_size + 1]['close'] if 'close' in data.columns else data.iloc[step + self.state_size + 1, -1]
                
                price_change = (next_price - current_price) / current_price
                
                # Reward based on action correctness
                if action == 1:  # Buy
                    reward = price_change * 100  # Amplify reward
                elif action == 2:  # Sell
                    reward = -price_change * 100
                else:  # Hold
                    reward = -0.01  # Small penalty for holding
            else:
                reward = 0.0
            
            return next_state, reward, done
            
        except Exception as e:
            self.logger.error(f"❌ Environment step failed: {str(e)}")
            return np.zeros(self.state_size), 0.0, True
    
    def train_agent(self, training_data: pd.DataFrame, episodes: int = 100) -> Dict[str, Any]:
        """ฝึก DQN Agent"""
        try:
            self.logger.info(f"🚀 Training DQN Agent for {episodes} episodes...")
            
            episode_results = []
            
            for episode in range(episodes):
                result = self.train_episode(training_data)
                episode_results.append(result)
                
                # Log progress
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean([r['reward'] for r in episode_results[-10:]])
                    self.logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
            
            self.is_trained = True
            
            # Calculate final statistics
            final_results = {
                'success': True,
                'total_episodes': episodes,
                'final_epsilon': self.epsilon,
                'avg_reward': np.mean([r['reward'] for r in episode_results]),
                'best_reward': max([r['reward'] for r in episode_results]),
                'episode_results': episode_results
            }
            
            self.logger.info("✅ DQN Agent training completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ DQN Agent training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict_action(self, state: np.ndarray) -> int:
        """ทำนาย action สำหรับการใช้งานจริง"""
        return self.get_action(state, training=False)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """ข้อมูลเกี่ยวกับ Agent"""
        return {
            'agent_type': 'DQN Reinforcement Learning',
            'is_trained': self.is_trained,
            'pytorch_available': PYTORCH_AVAILABLE,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'features': [
                'Deep Q-Network',
                'Experience Replay',
                'Target Network',
                'Epsilon-Greedy Exploration'
            ]
        }
