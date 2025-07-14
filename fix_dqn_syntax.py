#!/usr/bin/env python3
"""
🔧 FIX DQN SYNTAX ERROR
แก้ไข syntax error ใน DQN agent และเพิ่ม dynamic state size adjustment
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def fix_dqn_agent():
    """แก้ไข DQN agent โดยเพิ่ม dynamic state size adjustment"""
    
    dqn_file = Path("elliott_wave_modules/dqn_agent.py")
    
    if not dqn_file.exists():
        print(f"❌ File not found: {dqn_file}")
        return False
    
    # Read current file
    with open(dqn_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the __init__ method and add dynamic state size support
    init_fix = '''    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None, model_manager: Optional[Any] = None):
        self.config = config or {}
        self.component_name = "DQNReinforcementAgent"
        self.model_manager = model_manager  # Enterprise Model Manager integration
        
        # Initialize Advanced Terminal Logger
        if ADVANCED_LOGGING_AVAILABLE:
            try:
                self.logger = get_unified_logger()
                self.progress_manager = get_progress_manager()
                self.logger.info(f"🚀 {self.component_name} initialized with advanced logging")
            except Exception as e:
                self.logger = logger or get_unified_logger()
                self.progress_manager = None
                print(f"⚠️ Advanced logging failed, using fallback: {e}")
        else:
            self.logger = logger or get_unified_logger()
            self.progress_manager = None
        
        # DQN Parameters  
        self.state_size = self.config.get('dqn', {}).get('state_size', 20)  # Will be updated dynamically
        self.action_size = self.config.get('dqn', {}).get('action_size', 3)  # Hold, Buy, Sell
        self.learning_rate = self.config.get('dqn', {}).get('learning_rate', 0.001)
        self.gamma = self.config.get('dqn', {}).get('gamma', 0.95)
        self.epsilon = self.config.get('dqn', {}).get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('dqn', {}).get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('dqn', {}).get('epsilon_decay', 0.995)
        self.memory_size = self.config.get('dqn', {}).get('memory_size', 10000)
        
        # Network initialization flag
        self.networks_initialized = False
        self.actual_state_size = None
        
        # Initialize components
        try:
            self._initialize_components()
            
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
            raise'''
    
    # Dynamic state size method
    update_method = '''
    def update_state_size(self, new_state_size: int):
        """Update state size and reinitialize networks if needed"""
        try:
            if self.actual_state_size != new_state_size:
                self.actual_state_size = new_state_size
                old_state_size = self.state_size
                self.state_size = new_state_size
                
                # Reinitialize networks with correct state size
                self.logger.info(f"🔄 Updating DQN state_size from {old_state_size} to {new_state_size}")
                
                self.q_network = DQNNetwork(self.state_size, self.action_size)
                self.target_network = DQNNetwork(self.state_size, self.action_size)
                
                # Clear replay buffer to avoid dimension mismatches
                self.replay_buffer = ReplayBuffer(self.memory_size)
                
                # Reinitialize optimizer (PyTorch only)
                if PYTORCH_AVAILABLE:
                    self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                self.networks_initialized = True
                self.logger.info(f"✅ DQN networks reinitialized with state_size: {self.state_size}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update state size: {str(e)}")'''
    
    # Fixed get_action method with state size check
    get_action_fix = '''    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """เลือก action ด้วย epsilon-greedy policy"""
        try:
            # Check if state size matches network architecture and update if needed
            if len(state) != self.state_size:
                self.logger.warning(f"⚠️ State size mismatch: expected {self.state_size}, got {len(state)}")
                self.update_state_size(len(state))
            
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
            self.logger.error(f"State shape: {state.shape if hasattr(state, 'shape') else len(state)}, Expected state_size: {self.state_size}")
            return 0  # Default to Hold'''
    
    # Apply the fixes
    content_lines = content.split('\n')
    fixed_lines = []
    in_init_method = False
    in_get_action_method = False
    skip_lines = 0
    
    i = 0
    while i < len(content_lines):
        line = content_lines[i]
        
        if skip_lines > 0:
            skip_lines -= 1
            i += 1
            continue
        
        # Find and replace __init__ method
        if 'def __init__(self, config: Optional[Dict] = None' in line:
            fixed_lines.append(init_fix)
            # Skip the old __init__ method
            brace_count = 0
            j = i + 1
            while j < len(content_lines):
                next_line = content_lines[j]
                if next_line.strip().startswith('def ') and not next_line.strip().startswith('def __init__'):
                    break
                j += 1
            skip_lines = j - i - 1
            
        # Find and replace get_action method
        elif 'def get_action(self, state: np.ndarray' in line:
            fixed_lines.append(get_action_fix)
            # Skip the old get_action method
            j = i + 1
            while j < len(content_lines):
                next_line = content_lines[j]
                if next_line.strip().startswith('def ') and 'get_action' not in next_line:
                    break
                j += 1
            skip_lines = j - i - 1
            
        # Add update_state_size method after _initialize_components
        elif 'def _initialize_components(self):' in line:
            # Add the existing _initialize_components method
            fixed_lines.append(line)
            # Add lines until the end of this method
            j = i + 1
            while j < len(content_lines):
                method_line = content_lines[j]
                fixed_lines.append(method_line)
                if method_line.strip().startswith('def ') and j > i + 5:  # Next method
                    # Insert update_state_size before the next method
                    fixed_lines.insert(-1, update_method)
                    break
                j += 1
            skip_lines = j - i - 1
            
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content
    try:
        with open(dqn_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print(f"✅ DQN agent syntax fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to write fixed DQN file: {str(e)}")
        return False

def main():
    """Main function"""
    print("🔧 Fixing DQN Agent Syntax Error...")
    
    if fix_dqn_agent():
        print("🎉 DQN Agent syntax fixed successfully!")
        print("🚀 Ready to test dynamic state size adjustment!")
        return True
    else:
        print("❌ Failed to fix DQN Agent syntax")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 