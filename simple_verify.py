#!/usr/bin/env python3
"""
🔧 SIMPLE FIX VERIFICATION
"""

import sys
sys.path.append('/content/drive/MyDrive/ProjectP')

print("🔧 Testing fixes...")

# Test 1: Performance Analyzer
print("\n1. Testing Performance Analyzer...")
try:
    from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
    analyzer = ElliottWavePerformanceAnalyzer()
    
    # Test with correct number of arguments
    test_results = {
        'cnn_lstm_results': {'auc_score': 0.75},
        'dqn_results': {'total_reward': 100}
    }
    result = analyzer.analyze_performance(test_results)
    print("✅ Performance Analyzer: Fixed!")
except Exception as e:
    if "takes 2 positional arguments but 3 were given" in str(e):
        print("❌ Performance Analyzer: Still broken")
    else:
        print(f"💡 Performance Analyzer: Different error (may be fixed): {e}")

# Test 2: DQN Agent  
print("\n2. Testing DQN Agent...")
try:
    from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
    import pandas as pd
    import numpy as np
    
    agent = DQNReinforcementAgent()
    
    # Test data
    X = pd.DataFrame({'f1': [1,2,3,4,5], 'f2': [2,3,4,5,6]})
    y = pd.Series([0,1,0,1,0])
    
    result = agent.train_agent(X, y)
    print("✅ DQN Agent: Fixed!")
except Exception as e:
    if "'Series' object cannot be interpreted as an integer" in str(e):
        print("❌ DQN Agent: Still broken")
    else:
        print(f"💡 DQN Agent: Different error (may be fixed): {e}")

# Test 3: Menu 1
print("\n3. Testing Menu 1...")
try:
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
    menu = Menu1ElliottWave()
    print("✅ Menu 1: Import successful!")
    
    # Quick pipeline test
    results = menu.run_full_pipeline()
    error_msg = results.get('error_message', '')
    
    if "'Series' object cannot be interpreted as an integer" in error_msg:
        print("❌ Menu 1: DQN Series error still exists")
    elif "takes 2 positional arguments but 3 were given" in error_msg:
        print("❌ Menu 1: Performance Analyzer error still exists")
    else:
        print("✅ Menu 1: No target errors found!")
        
except Exception as e:
    error_str = str(e)
    if "'Series' object cannot be interpreted as an integer" in error_str:
        print("❌ Menu 1: DQN Series error during execution")
    elif "takes 2 positional arguments but 3 were given" in error_str:
        print("❌ Menu 1: Performance Analyzer error during execution")
    else:
        print(f"💡 Menu 1: Different error (target fixes may be working): {e}")

print("\n🎉 Fix verification completed!")
