#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 ENHANCED DQN AGENT WITH ENTERPRISE MODEL MANAGEMENT
Enhanced DQN Reinforcement Learning Agent พร้อมระบบจัดการโมเดลระดับ Enterprise

🎯 Enterprise Features:
✅ Enterprise Model Registration & Versioning
✅ Production-Ready RL Model Management
✅ Model Performance Tracking & Validation
✅ Automated Model Backup & Recovery
✅ Enterprise Compliance & Security
✅ RL Model Lifecycle Management

วันที่: 7 กรกฎาคม 2025
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
import json
import random
from collections import deque

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

# Add project root
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

# Import Enterprise Model Manager
try:
    from core.enterprise_model_manager_v2 import (
        EnterpriseModelManager, 
        ModelType, 
        ModelStatus,
        DeploymentStage
    )
    ENTERPRISE_MODEL_MANAGER_AVAILABLE = True
except ImportError:
    ENTERPRISE_MODEL_MANAGER_AVAILABLE = False

# TensorFlow imports
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

    
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class EnterpriseDQNAgent:
    """
    🏢 Enterprise DQN Reinforcement Learning Agent
    Production-Grade DQN with Enterprise Model Management
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 6,  # Enhanced actions: Buy_Small, Buy_Medium, Buy_Large, Sell_Small, Sell_Medium, Sell_Large
                 config: Dict[str, Any] = None,
                 logger: logging.Logger = None,
                 model_manager: EnterpriseModelManager = None):
        """Initialize Enterprise DQN Agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}
        self.logger = logger or get_unified_logger()
        
        # Initialize Model Manager
        if ENTERPRISE_MODEL_MANAGER_AVAILABLE:
            self.model_manager = model_manager or EnterpriseModelManager(
                config=self.config,
                logger=self.logger
            )
        else:
            self.model_manager = None
            self.logger.warning("⚠️ Enterprise Model Manager not available")
        
        # DQN Hyperparameters
        self.dqn_config = {
            'learning_rate': self.config.get('learning_rate', 0.001),
            'gamma': self.config.get('gamma', 0.95),  # Discount factor
            'epsilon_start': self.config.get('epsilon_start', 1.0),
            'epsilon_min': self.config.get('epsilon_min', 0.01),
            'epsilon_decay': self.config.get('epsilon_decay', 0.995),
            'memory_size': self.config.get('memory_size', 10000),
            'batch_size': self.config.get('batch_size', 32),
            'target_update_frequency': self.config.get('target_update_frequency', 100),
            'hidden_layers': self.config.get('hidden_layers', [256, 256, 128]),
            'dropout_rate': self.config.get('dropout_rate', 0.2)
        }
        
        # DQN State
        self.epsilon = self.dqn_config['epsilon_start']
        self.memory = deque(maxlen=self.dqn_config['memory_size'])
        self.q_network = None
        self.target_network = None
        self.current_model_id = None
        self.is_trained = False
        self.training_step = 0
        
        # Trading actions mapping
        self.action_mapping = {
            0: "Hold",
            1: "Buy_Small",
            2: "Buy_Medium", 
            3: "Buy_Large",
            4: "Sell_Small",
            5: "Sell_Medium",
            6: "Sell_Large"
        }
        
        # Enterprise settings
        self.enterprise_settings = {
            'target_reward': 0.70,
            'min_episodes': 100,
            'enable_model_versioning': True,
            'auto_backup': True,
            'performance_monitoring': True,
            'compliance_validation': True
        }
        
        # Initialize DQN networks
        self._build_networks()
        
        self.logger.info("🏢 Enterprise DQN Agent initialized")
    
    def _build_networks(self):
        """Build Q-Network and Target Network"""
        try:
            if not TENSORFLOW_AVAILABLE:
                raise Exception("TensorFlow not available")
            
            self.logger.info("🏗️ Building DQN neural networks...")
            
            # Main Q-Network
            self.q_network = self._create_network()
            
            # Target Network (copy of main network)
            self.target_network = self._create_network()
            
            # Initialize target network with same weights
            self.target_network.set_weights(self.q_network.get_weights())
            
            self.logger.info("✅ DQN networks created successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build DQN networks: {e}")
            raise
    
    def _create_network(self) -> 'tf.keras.Model':
        """Create DQN neural network architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.dqn_config['hidden_layers'][0], 
                       input_dim=self.state_size, 
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dqn_config['dropout_rate']))
        
        # Hidden layers
        for units in self.dqn_config['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dqn_config['dropout_rate']))
        
        # Output layer (Q-values for each action)
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.dqn_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, exploration: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            exploration: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Action index
        """
        try:
            if exploration and np.random.random() <= self.epsilon:
                # Random action (exploration)
                action = random.randrange(self.action_size)
                self.logger.debug(f"🎲 Random action: {self.action_mapping.get(action, action)}")
                return action
            
            # Predict Q-values
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            q_values = self.q_network.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            
            self.logger.debug(f"🧠 Q-Network action: {self.action_mapping.get(action, action)}")
            return action
            
        except Exception as e:
            self.logger.error(f"❌ Action selection failed: {e}")
            return 0  # Default to Hold
    
    def replay(self, batch_size: int = None) -> Dict[str, float]:
        """
        Train the Q-Network using experience replay
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Training metrics
        """
        try:
            if len(self.memory) < (batch_size or self.dqn_config['batch_size']):
                return {}
            
            batch_size = batch_size or self.dqn_config['batch_size']
            
            # Sample random batch from memory
            batch = random.sample(self.memory, batch_size)
            
            states = np.array([experience[0] for experience in batch])
            actions = np.array([experience[1] for experience in batch])
            rewards = np.array([experience[2] for experience in batch])
            next_states = np.array([experience[3] for experience in batch])
            dones = np.array([experience[4] for experience in batch])
            
            # Current Q-values
            current_q_values = self.q_network.predict(states, verbose=0)
            
            # Next Q-values from target network
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            # Calculate target Q-values using Bellman equation
            target_q_values = current_q_values.copy()
            
            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.dqn_config['gamma'] * np.max(next_q_values[i])
            
            # Train the Q-network
            history = self.q_network.fit(states, target_q_values, 
                                       epochs=1, verbose=0, batch_size=batch_size)
            
            # Decay epsilon
            if self.epsilon > self.dqn_config['epsilon_min']:
                self.epsilon *= self.dqn_config['epsilon_decay']
            
            # Update target network periodically
            self.training_step += 1
            if self.training_step % self.dqn_config['target_update_frequency'] == 0:
                self.update_target_network()
            
            return {
                'loss': float(history.history['loss'][0]),
                'mae': float(history.history['mae'][0]),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }
            
        except Exception as e:
            self.logger.error(f"❌ DQN replay training failed: {e}")
            return {}
    
    def update_target_network(self):
        """Update target network with weights from main network"""
        try:
            self.target_network.set_weights(self.q_network.get_weights())
            self.logger.debug("🎯 Target network updated")
        except Exception as e:
            self.logger.error(f"❌ Failed to update target network: {e}")
    
    def train_agent(self, 
                   training_environment: Any,
                   episodes: int,
                   model_name: str = "DQN_Elliott_Wave_Agent",
                   business_purpose: str = "Elliott Wave Trading Strategy") -> Dict[str, Any]:
        """
        Train DQN agent with enterprise management
        
        Args:
            training_environment: Trading environment for training
            episodes: Number of training episodes
            model_name: Name for the model
            business_purpose: Business purpose description
            
        Returns:
            Training results and metrics
        """
        try:
            self.logger.info("🚀 Starting Enterprise DQN Agent Training...")
            
            # Register new model in enterprise system
            if self.model_manager:
                self.current_model_id = self.model_manager.register_new_model(
                    model_name=model_name,
                    model_type=ModelType.DQN_AGENT,
                    business_purpose=business_purpose,
                    use_case_description="Deep Q-Network agent for Elliott Wave-based trading decisions",
                    training_config=self.dqn_config
                )
                self.logger.info(f"📝 Model registered: {self.current_model_id}")
            
            # Update model status to training
            if self.model_manager:
                self.model_manager.update_model_status(
                    self.current_model_id, 
                    ModelStatus.TRAINING
                )
            
            training_start_time = datetime.now()
            
            # Training metrics
            episode_rewards = []
            episode_losses = []
            episode_actions = []
            best_reward = float('-inf')
            
            for episode in range(episodes):
                state = training_environment.reset()
                total_reward = 0
                episode_loss = 0
                actions_taken = []
                done = False
                step = 0
                
                while not done and step < 1000:  # Max steps per episode
                    # Choose action
                    action = self.act(state)
                    actions_taken.append(action)
                    
                    # Take action in environment
                    next_state, reward, done, info = training_environment.step(action)
                    
                    # Store experience
                    self.remember(state, action, reward, next_state, done)
                    
                    # Train if enough experiences
                    if len(self.memory) > self.dqn_config['batch_size']:
                        training_metrics = self.replay()
                        if training_metrics:
                            episode_loss += training_metrics.get('loss', 0)
                    
                    state = next_state
                    total_reward += reward
                    step += 1
                
                episode_rewards.append(total_reward)
                episode_losses.append(episode_loss / max(step, 1))
                episode_actions.append(actions_taken)
                
                # Track best performance
                if total_reward > best_reward:
                    best_reward = total_reward
                
                # Log progress
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    self.logger.info(f"Episode {episode}/{episodes}: Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            # Calculate training metrics
            final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            training_metrics = {
                'final_average_reward': final_avg_reward,
                'best_reward': best_reward,
                'total_episodes': episodes,
                'average_loss': np.mean(episode_losses),
                'final_epsilon': self.epsilon,
                'convergence_episode': self._find_convergence_episode(episode_rewards),
                'action_distribution': self._calculate_action_distribution(episode_actions)
            }
            
            validation_metrics = {
                'reward_stability': np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards),
                'performance_score': final_avg_reward,
                'exploration_ratio': self.epsilon,
                'training_efficiency': final_avg_reward / episodes if episodes > 0 else 0
            }
            
            # Update model with training results
            if self.model_manager:
                model_file_path = self.model_manager.save_trained_model(
                    model_id=self.current_model_id,
                    model_object={
                        'q_network': self.q_network,
                        'target_network': self.target_network,
                        'memory': list(self.memory),
                        'epsilon': self.epsilon,
                        'action_mapping': self.action_mapping
                    },
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    training_duration=training_duration,
                    training_samples=len(self.memory),
                    validation_samples=0,
                    feature_count=self.state_size
                )
                
                self.logger.info(f"💾 DQN Agent saved: {model_file_path}")
            
            # Mark as trained
            self.is_trained = True
            
            # Prepare results
            results = {
                'model_id': self.current_model_id,
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics,
                'training_duration_seconds': training_duration,
                'total_episodes': episodes,
                'memory_size': len(self.memory),
                'state_size': self.state_size,
                'action_size': self.action_size,
                'enterprise_compliant': final_avg_reward >= self.enterprise_settings['target_reward']
            }
            
            # Log training completion
            if final_avg_reward >= self.enterprise_settings['target_reward']:
                self.logger.info(f"🎉 DQN training completed successfully! Avg Reward: {final_avg_reward:.4f}")
            else:
                self.logger.warning(f"⚠️ DQN training completed but performance below target: {final_avg_reward:.4f}")
            
            return results
            
        except Exception as e:
            # Update model status to failed
            if self.model_manager and self.current_model_id:
                self.model_manager.update_model_status(
                    self.current_model_id, 
                    ModelStatus.FAILED
                )
            
            self.logger.error(f"❌ DQN training failed: {e}")
            raise
    
    def _find_convergence_episode(self, rewards: List[float]) -> int:
        """Find episode where rewards started converging"""
        if len(rewards) < 50:
            return len(rewards)
        
        window_size = 20
        for i in range(window_size, len(rewards)):
            recent_std = np.std(rewards[i-window_size:i])
            if recent_std < 0.1:  # Convergence threshold
                return i
        
        return len(rewards)
    
    def _calculate_action_distribution(self, episode_actions: List[List[int]]) -> Dict[str, float]:
        """Calculate distribution of actions taken during training"""
        all_actions = [action for episode in episode_actions for action in episode]
        
        if not all_actions:
            return {}
        
        action_counts = {}
        for action in all_actions:
            action_name = self.action_mapping.get(action, f"Action_{action}")
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        total_actions = len(all_actions)
        return {action: count/total_actions for action, count in action_counts.items()}
    
    def predict_action(self, state: np.ndarray) -> Tuple[int, str, float]:
        """
        Predict best action for given state
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action_index, action_name, confidence)
        """
        try:
            if not self.is_trained or self.q_network is None:
                raise Exception("Agent is not trained")
            
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            q_values = self.q_network.predict(state, verbose=0)
            action_index = np.argmax(q_values[0])
            action_name = self.action_mapping.get(action_index, f"Action_{action_index}")
            confidence = float(np.max(q_values[0]))
            
            return action_index, action_name, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Action prediction failed: {e}")
            return 0, "Hold", 0.0
    
    def load_agent_from_enterprise(self, model_id: str) -> bool:
        """
        Load DQN agent from enterprise system
        
        Args:
            model_id: Enterprise model identifier
            
        Returns:
            Success status
        """
        try:
            if not self.model_manager:
                raise Exception("Enterprise Model Manager not available")
            
            self.logger.info(f"📥 Loading DQN agent from enterprise system: {model_id}")
            
            # Load model from enterprise system
            model_data = self.model_manager.load_model(model_id)
            
            agent_data = model_data['model']
            self.q_network = agent_data['q_network']
            self.target_network = agent_data['target_network']
            self.epsilon = agent_data['epsilon']
            self.action_mapping = agent_data['action_mapping']
            
            # Restore memory if available
            if 'memory' in agent_data:
                self.memory = deque(agent_data['memory'], maxlen=self.dqn_config['memory_size'])
            
            self.current_model_id = model_id
            self.is_trained = True
            
            self.logger.info(f"✅ DQN agent loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load DQN agent: {e}")
            return False
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent performance summary"""
        try:
            if not self.model_manager or not self.current_model_id:
                return {"error": "No agent loaded or model manager not available"}
            
            return self.model_manager.get_model_performance_summary(self.current_model_id)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance summary: {e}")
            return {"error": str(e)}

# Factory function
def create_enterprise_dqn_agent(state_size: int,
                               action_size: int = 6,
                               config: Dict[str, Any] = None,
                               logger: logging.Logger = None) -> EnterpriseDQNAgent:
    """Create Enterprise DQN Agent instance"""
    return EnterpriseDQNAgent(
        state_size=state_size,
        action_size=action_size,
        config=config,
        logger=logger
    )

# Export
__all__ = ['EnterpriseDQNAgent', 'create_enterprise_dqn_agent']
