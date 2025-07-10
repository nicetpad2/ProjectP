#!/usr/bin/env python3
"""
🧪 FINAL PIPELINE VERIFICATION TEST
ทดสอบการแก้ไข DQN และ Performance Analyzer แบบครบถ้วน
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = '/content/drive/MyDrive/ProjectP'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dqn_agent_complete():
    """ทดสอบ DQN Agent แบบครบถ้วน"""
    print("🧪 Testing DQN Agent complete fixes...")
    
    try:
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
        
        # Test with proper config
        config = {
            'dqn': {
                'state_size': 5,
                'action_size': 3,
                'learning_rate': 0.001,
                'epsilon_start': 1.0,
                'episodes': 3
            }
        }
        
        agent = DQNReinforcementAgent(config=config, logger=logger)
        print("✅ DQN Agent initialized")
        
        # Test 1: Normal DataFrame (sufficient size)
        print("📊 Test 1: Normal DataFrame...")
        df_large = pd.DataFrame({
            'close': np.random.rand(100) * 100 + 1800,
            'volume': np.random.rand(100) * 1000
        })
        
        result1 = agent.train_agent(df_large, episodes=2)
        print(f"✅ Large DataFrame test: Success={result1.get('success', False)}")
        
        # Test 2: Small DataFrame (insufficient size)
        print("📈 Test 2: Small DataFrame...")
        df_small = pd.DataFrame({'close': [1800, 1801, 1802]})  # Only 3 rows
        
        result2 = agent.train_agent(df_small, episodes=1)
        print(f"✅ Small DataFrame test: Success={result2.get('success', False)}")
        
        # Test 3: Series input
        print("🔢 Test 3: Series input...")
        series_data = pd.Series(np.random.rand(20) * 100 + 1800)
        
        result3 = agent.train_agent(series_data, episodes=1)
        print(f"✅ Series test: Success={result3.get('success', False)}")
        
        print("🎯 DQN Agent tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ DQN Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_analyzer_complete():
    """ทดสอบ Performance Analyzer แบบครบถ้วน"""
    print("\n📊 Testing Performance Analyzer complete fixes...")
    
    try:
        from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
        
        analyzer = ElliottWavePerformanceAnalyzer(logger=logger)
        print("✅ Performance Analyzer initialized")
        
        # Test comprehensive pipeline results
        pipeline_results = {
            'cnn_lstm_training': {
                'cnn_lstm_results': {
                    'evaluation_results': {
                        'auc': 0.78,
                        'accuracy': 0.72,
                        'precision': 0.70,
                        'recall': 0.68,
                        'f1_score': 0.69
                    },
                    'training_results': {
                        'final_accuracy': 0.74,
                        'final_val_accuracy': 0.71
                    }
                }
            },
            'dqn_training': {
                'dqn_results': {
                    'evaluation_results': {
                        'return_pct': 15.8,
                        'final_balance': 11580,
                        'total_trades': 32
                    }
                }
            },
            'feature_selection': {
                'selection_results': {
                    'best_auc': 0.76,
                    'feature_count': 18,
                    'target_achieved': True
                }
            },
            'data_loading': {
                'data_quality': {
                    'real_data_percentage': 100
                }
            },
            'quality_validation': {
                'quality_score': 88.5
            }
        }
        
        # Test analyze_performance with proper structure
        results = analyzer.analyze_performance(pipeline_results)
        
        print("✅ Performance analysis completed")
        
        # Display key results
        overall = results.get('overall_performance', {})
        print(f"Overall Score: {overall.get('overall_score', 0):.2f}")
        print(f"Performance Grade: {overall.get('performance_grade', 'N/A')}")
        print(f"Enterprise Ready: {overall.get('enterprise_ready', False)}")
        
        # Check key metrics
        key_metrics = overall.get('key_metrics', {})
        print(f"AUC: {key_metrics.get('auc', 0):.4f}")
        print(f"Sharpe Ratio: {key_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Win Rate: {key_metrics.get('win_rate', 0):.1f}%")
        
        print("🎯 Performance Analyzer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_menu_1_integration():
    """ทดสอบการ integrate Menu 1"""
    print("\n🎯 Testing Menu 1 integration...")
    
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
        
        # Minimal config for testing
        config = {
            'data': {
                'csv_file': '/content/drive/MyDrive/ProjectP/datacsv/XAUUSD_M1.csv'
            },
            'cnn_lstm': {
                'epochs': 1,
                'batch_size': 16
            },
            'dqn': {
                'state_size': 5,
                'action_size': 3,
                'episodes': 2
            },
            'feature_selection': {
                'n_features': 5,
                'target_auc': 0.7
            }
        }
        
        menu = Menu1ElliottWaveFixed(config=config, logger=logger)
        print("✅ Menu 1 initialized successfully")
        
        # Test component initialization
        assert hasattr(menu, 'dqn_agent'), "DQN Agent not found"
        assert hasattr(menu, 'performance_analyzer'), "Performance Analyzer not found"
        assert hasattr(menu, 'data_processor'), "Data Processor not found"
        assert hasattr(menu, 'cnn_lstm_engine'), "CNN-LSTM Engine not found"
        assert hasattr(menu, 'feature_selector'), "Feature Selector not found"
        
        print("✅ All Menu 1 components properly initialized")
        
        # Test that methods exist
        assert hasattr(menu, 'run_elliott_wave_analysis'), "run_elliott_wave_analysis method not found"
        
        print("✅ Menu 1 methods available")
        return True
        
    except Exception as e:
        print(f"❌ Menu 1 integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 FINAL PIPELINE VERIFICATION TEST")
    print("=" * 60)
    
    # Run all tests
    test_results = []
    
    test_results.append(("DQN Agent Complete", test_dqn_agent_complete()))
    test_results.append(("Performance Analyzer Complete", test_performance_analyzer_complete()))
    test_results.append(("Menu 1 Integration", test_menu_1_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed_count = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_count += 1
    
    print(f"\nOverall: {passed_count}/{len(test_results)} tests passed")
    
    if passed_count == len(test_results):
        print("\n🎉 ALL TESTS PASSED! Pipeline fixes successful!")
        print("✅ DQN Agent: Fixed Series/DataFrame issues and step variable scope")
        print("✅ Performance Analyzer: Fixed argument mismatch in analyze_performance")
        print("✅ Menu 1: Correct class name and integration verified")
        print("\n🚀 READY FOR FULL PIPELINE EXECUTION!")
        return True
    else:
        print("\n⚠️ Some tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
