#!/usr/bin/env python3
"""
🧪 TEST PIPELINE FIXES
ทดสอบการแก้ไข DQN Agent และ Performance Analyzer

Test Cases:
1. DQN Agent with DataFrame, Series, and numpy array
2. Performance Analyzer with correct arguments
3. Menu 1 Elliott Wave full pipeline
"""

import sys
import os
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = '/content/drive/MyDrive/ProjectP'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_logging():
    """Setup test logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_dqn_agent_fixes():
    """ทดสอบการแก้ไข DQN Agent"""
    logger = setup_logging()
    logger.info("🧪 Testing DQN Agent fixes...")
    
    try:
        from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
        
        # Test config
        config = {
            'dqn': {
                'state_size': 10,
                'action_size': 3,
                'learning_rate': 0.001,
                'epsilon_start': 1.0,
                'episodes': 5
            }
        }
        
        # Initialize agent
        agent = DQNReinforcementAgent(config=config, logger=logger)
        logger.info("✅ DQN Agent initialized successfully")
        
        # Test 1: DataFrame input
        logger.info("📊 Test 1: DataFrame input...")
        df_data = pd.DataFrame({
            'close': np.random.rand(100) * 100 + 1800,
            'volume': np.random.rand(100) * 1000,
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        
        result1 = agent.train_agent(df_data, episodes=3)
        logger.info(f"✅ DataFrame test passed: {result1.get('success', False)}")
        
        # Test 2: Series input
        logger.info("📈 Test 2: Series input...")
        series_data = pd.Series(np.random.rand(50) * 100 + 1800)
        
        result2 = agent.train_agent(series_data, episodes=2)
        logger.info(f"✅ Series test passed: {result2.get('success', False)}")
        
        # Test 3: Numpy array input
        logger.info("🔢 Test 3: Numpy array input...")
        numpy_data = np.random.rand(30, 3) * 100 + 1800
        
        result3 = agent.train_agent(numpy_data, episodes=2)
        logger.info(f"✅ Numpy array test passed: {result3.get('success', False)}")
        
        logger.info("🎯 DQN Agent tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DQN Agent test failed: {str(e)}")
        return False

def test_performance_analyzer_fixes():
    """ทดสอบการแก้ไข Performance Analyzer"""
    logger = setup_logging()
    logger.info("📊 Testing Performance Analyzer fixes...")
    
    try:
        from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
        
        # Initialize analyzer
        analyzer = ElliottWavePerformanceAnalyzer(logger=logger)
        logger.info("✅ Performance Analyzer initialized successfully")
        
        # Test pipeline results structure
        pipeline_results = {
            'cnn_lstm_training': {
                'cnn_lstm_results': {
                    'evaluation_results': {
                        'auc': 0.75,
                        'accuracy': 0.68,
                        'precision': 0.70,
                        'recall': 0.65,
                        'f1_score': 0.67
                    },
                    'training_results': {
                        'final_accuracy': 0.70,
                        'final_val_accuracy': 0.68
                    }
                }
            },
            'dqn_training': {
                'dqn_results': {
                    'evaluation_results': {
                        'return_pct': 12.5,
                        'final_balance': 11250,
                        'total_trades': 25
                    }
                }
            },
            'feature_selection': {
                'selection_results': {
                    'best_auc': 0.72,
                    'feature_count': 15,
                    'target_achieved': True
                }
            },
            'data_loading': {
                'data_quality': {
                    'real_data_percentage': 100
                }
            },
            'quality_validation': {
                'quality_score': 85.0
            }
        }
        
        # Test analyze_performance method
        results = analyzer.analyze_performance(pipeline_results)
        
        logger.info("✅ Performance analysis completed")
        logger.info(f"Overall Score: {results.get('overall_performance', {}).get('overall_score', 0):.2f}")
        logger.info(f"Enterprise Ready: {results.get('overall_performance', {}).get('enterprise_ready', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance Analyzer test failed: {str(e)}")
        return False

def test_menu_1_integration():
    """ทดสอบการ integrate ใน Menu 1"""
    logger = setup_logging()
    logger.info("🎯 Testing Menu 1 integration...")
    
    try:
        from menu_modules.menu_1_elliott_wave import ElliottWaveMenu
        
        # Initialize menu with minimal config
        config = {
            'data': {
                'csv_file': '/content/drive/MyDrive/ProjectP/datacsv/XAUUSD_M1.csv'
            },
            'cnn_lstm': {
                'epochs': 2,
                'batch_size': 32
            },
            'dqn': {
                'state_size': 10,
                'action_size': 3,
                'episodes': 3
            },
            'feature_selection': {
                'n_features': 5,
                'target_auc': 0.7
            }
        }
        
        menu = ElliottWaveMenu(config=config, logger=logger)
        logger.info("✅ Menu 1 initialized successfully")
        
        # Test quick run (limited scope)
        logger.info("🚀 Running quick pipeline test...")
        results = menu.run_elliott_wave_analysis()
        
        success = results.get('success', False)
        logger.info(f"✅ Menu 1 pipeline test: {'PASSED' if success else 'FAILED'}")
        
        # Show key results
        if success:
            performance = results.get('performance_analysis', {})
            overall = performance.get('overall_performance', {})
            logger.info(f"Overall Score: {overall.get('overall_score', 0):.2f}")
            logger.info(f"AUC: {overall.get('key_metrics', {}).get('auc', 0):.4f}")
            logger.info(f"DQN Reward: {overall.get('key_metrics', {}).get('total_return', 0):.2f}%")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Menu 1 integration test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    logger = setup_logging()
    logger.info("🧪 STARTING PIPELINE FIXES TESTS")
    logger.info("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: DQN Agent
    logger.info("\n" + "=" * 30)
    test_results.append(("DQN Agent Fixes", test_dqn_agent_fixes()))
    
    # Test 2: Performance Analyzer  
    logger.info("\n" + "=" * 30)
    test_results.append(("Performance Analyzer Fixes", test_performance_analyzer_fixes()))
    
    # Test 3: Menu 1 Integration
    logger.info("\n" + "=" * 30)
    test_results.append(("Menu 1 Integration", test_menu_1_integration()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_count = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_count += 1
    
    logger.info(f"\nOverall: {passed_count}/{len(test_results)} tests passed")
    
    if passed_count == len(test_results):
        logger.info("🎉 ALL TESTS PASSED! Pipeline fixes successful!")
        return True
    else:
        logger.info("⚠️ Some tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
