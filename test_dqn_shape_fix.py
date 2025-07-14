#!/usr/bin/env python3
"""
🧪 TEST DQN SHAPE FIX
ทดสอบการแก้ไขปัญหา DQN shape mismatch error
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_dqn_dynamic_state_size():
    """ทดสอบ DQN agent dynamic state size adjustment"""
    
    print("🧪 Testing DQN Dynamic State Size Adjustment...")
    
    try:
        # Import required modules
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
        from core.unified_enterprise_logger import get_unified_logger
        
        # Initialize logger
        logger = get_unified_logger()
        
        # Create DQN config
        dqn_config = {
            'dqn': {
                'state_size': 20,  # Start with different size
                'action_size': 3,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 1000
            }
        }
        
        # Initialize DQN agent
        dqn_agent = DQNReinforcementAgent(config=dqn_config, logger=logger)
        
        print(f"✅ DQN Agent initialized with initial state_size: {dqn_agent.state_size}")
        
        # Test with different state sizes
        test_cases = [
            ("Small state", np.random.randn(15)),
            ("Medium state", np.random.randn(25)),
            ("Large state", np.random.randn(30)),
            ("Extra large state", np.random.randn(35))
        ]
        
        for test_name, test_state in test_cases:
            print(f"\n🔄 Testing {test_name} (size: {len(test_state)})...")
            
            try:
                # This should automatically update state_size if needed
                action = dqn_agent.get_action(test_state, training=False)
                
                print(f"✅ {test_name}: Action {action} selected successfully")
                print(f"   Current DQN state_size: {dqn_agent.state_size}")
                
            except Exception as e:
                print(f"❌ {test_name} failed: {str(e)}")
                return False
        
        # Test with common feature selection output (around 30 features)
        print(f"\n🔄 Testing with typical feature selection output (30 features)...")
        typical_features = np.random.randn(30)
        
        try:
            action = dqn_agent.get_action(typical_features, training=False)
            print(f"✅ 30-feature test: Action {action} selected successfully")
            print(f"   Final DQN state_size: {dqn_agent.state_size}")
            
        except Exception as e:
            print(f"❌ 30-feature test failed: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ DQN dynamic state size test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dqn_with_feature_selection():
    """ทดสอบ DQN integration กับ feature selection"""
    
    print("\n🔄 Testing DQN with Feature Selection Integration...")
    
    try:
        # Import required modules
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
        from elliott_wave_modules.feature_engineering import ElliottWaveFeatureEngineer
        from core.unified_enterprise_logger import get_unified_logger
        
        # Initialize logger
        logger = get_unified_logger()
        
        # Create sample data similar to real scenario
        dates = pd.date_range('2023-01-01', periods=200, freq='1min')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1800, 2000, 200),
            'high': np.random.uniform(1810, 2010, 200),
            'low': np.random.uniform(1790, 1990, 200),
            'close': np.random.uniform(1800, 2000, 200),
            'volume': np.random.uniform(100, 1000, 200)
        })
        
        print("✅ Sample market data created")
        
        # Initialize feature engineer
        feature_engineer = ElliottWaveFeatureEngineer(
            config={}, 
            logger=logger
        )
        
        # Create features (this will create multiple features)
        features_df = feature_engineer.create_all_features(sample_data)
        
        print(f"✅ Features created: {features_df.shape[1]} features")
        
        # Prepare ML data (this includes target creation)
        X, y = feature_engineer.prepare_ml_data(features_df)
        
        print(f"✅ ML data prepared: X shape {X.shape}, y shape {y.shape}")
        
        # Initialize DQN agent with initial guess
        dqn_config = {
            'dqn': {
                'state_size': 20,  # Will be updated
                'action_size': 3,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_start': 0.1,  # Lower epsilon for testing
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 1000
            }
        }
        
        dqn_agent = DQNReinforcementAgent(config=dqn_config, logger=logger)
        
        print(f"✅ DQN Agent initialized with state_size: {dqn_agent.state_size}")
        
        # Test with actual feature data
        if len(X) > 0:
            test_state = X.iloc[0].values  # Get first row as state
            print(f"🔄 Testing with real features (size: {len(test_state)})...")
            
            action = dqn_agent.get_action(test_state, training=False)
            
            print(f"✅ Real features test: Action {action} selected successfully")
            print(f"   Updated DQN state_size: {dqn_agent.state_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ DQN feature selection integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_menu5_dqn_integration():
    """ทดสอบ Menu 5 DQN integration"""
    
    print("\n🔄 Testing Menu 5 DQN Integration...")
    
    try:
        # Import Menu 5 system
        from menu_modules.menu_5_oms_mm_100usd import Menu5OMSMMSystem
        
        # Initialize Menu 5
        menu5 = Menu5OMSMMSystem()
        
        # Load Menu 1 strategy (this should use updated DQN config)
        success = menu5.load_menu1_strategy()
        
        if success:
            print("✅ Menu 5 loaded Menu 1 strategy successfully")
            print(f"   DQN Agent state_size: {menu5.dqn_agent.state_size}")
            
            # Test basic DQN functionality
            test_state = np.random.randn(25)  # Different size to test dynamic adjustment
            
            action = menu5.dqn_agent.get_action(test_state, training=False)
            print(f"✅ Menu 5 DQN test: Action {action} selected successfully")
            print(f"   Updated state_size: {menu5.dqn_agent.state_size}")
            
        else:
            print("❌ Menu 5 failed to load Menu 1 strategy")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Menu 5 DQN integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎯 DQN SHAPE MISMATCH FIX VERIFICATION")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: DQN dynamic state size
    if test_dqn_dynamic_state_size():
        success_count += 1
        print("✅ Test 1: DQN Dynamic State Size - PASSED")
    else:
        print("❌ Test 1: DQN Dynamic State Size - FAILED")
    
    # Test 2: DQN with feature selection
    if test_dqn_with_feature_selection():
        success_count += 1
        print("✅ Test 2: DQN Feature Selection Integration - PASSED")
    else:
        print("❌ Test 2: DQN Feature Selection Integration - FAILED")
    
    # Test 3: Menu 5 DQN integration
    if test_menu5_dqn_integration():
        success_count += 1
        print("✅ Test 3: Menu 5 DQN Integration - PASSED")
    else:
        print("❌ Test 3: Menu 5 DQN Integration - FAILED")
    
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 DQN SHAPE MISMATCH FIX VERIFICATION COMPLETE - ALL TESTS PASSED!")
        print("🚀 The mat1 and mat2 shapes error should be resolved!")
        return True
    else:
        print("❌ DQN SHAPE MISMATCH FIX VERIFICATION INCOMPLETE - SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 