#!/usr/bin/env python3
"""
🧪 QUICK PIPELINE TEST
ทดสอบการแก้ไข DQN และ Performance Analyzer แบบรวดเร็ว
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np

# Add project root to path
project_root = '/content/drive/MyDrive/ProjectP'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("🧪 Testing DQN Agent fixes...")

try:
    # Test DQN Agent
    from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
    
    config = {'dqn': {'state_size': 5, 'action_size': 3}}
    agent = DQNReinforcementAgent(config=config)
    
    # Test with DataFrame
    df_data = pd.DataFrame({'close': np.random.rand(20) * 100 + 1800})
    result = agent.train_agent(df_data, episodes=2)
    print(f"✅ DQN DataFrame test: {result.get('success', False)}")
    
    # Test with Series  
    series_data = pd.Series(np.random.rand(15) * 100 + 1800)
    result = agent.train_agent(series_data, episodes=1)
    print(f"✅ DQN Series test: {result.get('success', False)}")
    
except Exception as e:
    print(f"❌ DQN test failed: {str(e)}")

print("\n📊 Testing Performance Analyzer fixes...")

try:
    # Test Performance Analyzer
    from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
    
    analyzer = ElliottWavePerformanceAnalyzer()
    
    # Test with proper pipeline results structure
    pipeline_results = {
        'cnn_lstm_training': {'cnn_lstm_results': {'evaluation_results': {'auc': 0.75}}},
        'dqn_training': {'dqn_results': {'evaluation_results': {'return_pct': 10.0}}},
        'feature_selection': {'selection_results': {'best_auc': 0.70}},
        'data_loading': {'data_quality': {'real_data_percentage': 100}},
        'quality_validation': {'quality_score': 85.0}
    }
    
    results = analyzer.analyze_performance(pipeline_results)
    print(f"✅ Performance Analyzer test: {results is not None}")
    print(f"Overall Score: {results.get('overall_performance', {}).get('overall_score', 0):.2f}")
    
except Exception as e:
    print(f"❌ Performance Analyzer test failed: {str(e)}")

print("\n🎯 Testing Menu 1 quick integration...")

try:
    # Test Menu 1 pipeline structure
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
    
    config = {
        'data': {'csv_file': '/content/drive/MyDrive/ProjectP/datacsv/XAUUSD_M1.csv'},
        'cnn_lstm': {'epochs': 1, 'batch_size': 16},
        'dqn': {'episodes': 2},
        'feature_selection': {'n_features': 3}
    }
    
    menu = Menu1ElliottWaveFixed(config=config)
    print("✅ Menu 1 initialization successful")
    
    # Test pipeline components initialization
    assert hasattr(menu, 'dqn_agent'), "DQN Agent not initialized"
    assert hasattr(menu, 'performance_analyzer'), "Performance Analyzer not initialized"
    print("✅ Menu 1 components properly initialized")
    
except Exception as e:
    print(f"❌ Menu 1 test failed: {str(e)}")

print("\n🎉 Quick tests completed!")
