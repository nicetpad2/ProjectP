#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED ELLIOTT WAVE SYSTEM INTEGRATION TEST
Test script for the Enhanced Elliott Wave System with Advanced Multi-timeframe Analysis

This script validates:
1. Advanced Elliott Wave Analyzer integration
2. Enhanced DQN Agent with Elliott Wave rewards
3. Multi-timeframe wave analysis
4. Curriculum learning functionality
5. Trading recommendation generation
6. System integration and data flow
"""

import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import warnings

# Essential imports
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import test components
try:
    from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
    ENHANCED_MENU_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced Menu not available: {e}")
    ENHANCED_MENU_AVAILABLE = False

try:
    from elliott_wave_modules.advanced_elliott_wave_analyzer import AdvancedElliottWaveAnalyzer
    ADVANCED_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced Elliott Wave Analyzer not available: {e}")
    ADVANCED_ANALYZER_AVAILABLE = False

try:
    from elliott_wave_modules.enhanced_dqn_agent import EnhancedDQNAgent
    ENHANCED_DQN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced DQN Agent not available: {e}")
    ENHANCED_DQN_AVAILABLE = False


class EnhancedElliottWaveIntegrationTest:
    """Comprehensive integration test for Enhanced Elliott Wave System"""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = None
        self.start_time = time.time()
        
    def generate_test_data(self, n_samples=1000):
        """Generate realistic test market data"""
        print("📊 Generating test market data...")
        
        # Generate realistic OHLC data with trends and patterns
        np.random.seed(42)
        
        # Base price with trend
        base_price = 1800.0
        trend = np.linspace(0, 50, n_samples)  # Upward trend
        noise = np.random.normal(0, 5, n_samples)
        
        # Create Elliott Wave-like pattern
        wave_pattern = 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + \
                      5 * np.sin(np.linspace(0, 8*np.pi, n_samples))
        
        # Combine to create realistic price movement
        close_prices = base_price + trend + noise + wave_pattern
        
        # Generate OHLC from close prices
        data = []
        for i in range(n_samples):
            close = close_prices[i]
            daily_range = abs(np.random.normal(0, 2))
            
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            open_price = low + np.random.uniform(0, high - low)
            
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=n_samples-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        self.test_data = pd.DataFrame(data)
        self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
        self.test_data.set_index('timestamp', inplace=True)
        
        print(f"✅ Generated {len(self.test_data)} rows of test data")
        print(f"   Price range: {self.test_data['close'].min():.2f} - {self.test_data['close'].max():.2f}")
        
        return self.test_data
    
    def test_advanced_elliott_wave_analyzer(self):
        """Test Advanced Elliott Wave Analyzer independently"""
        print("\n🌊 Testing Advanced Elliott Wave Analyzer...")
        
        if not ADVANCED_ANALYZER_AVAILABLE:
            print("⚠️ Advanced Elliott Wave Analyzer not available - skipping test")
            self.test_results['advanced_analyzer'] = {'status': 'skipped', 'reason': 'not_available'}
            return False
        
        try:
            # Initialize analyzer
            analyzer = AdvancedElliottWaveAnalyzer(
                timeframes=['1m', '5m', '15m'],
                logger=None
            )
            
            # Test multi-timeframe analysis
            print("   Testing multi-timeframe wave analysis...")
            analysis_results = analyzer.analyze_multi_timeframe_waves(self.test_data)
            
            if analysis_results and len(analysis_results) > 0:
                print(f"   ✅ Multi-timeframe analysis successful: {len(analysis_results)} timeframes analyzed")
                
                # Test feature extraction
                print("   Testing wave feature extraction...")
                wave_features = analyzer.extract_wave_features(analysis_results)
                
                if wave_features is not None and len(wave_features) > 0:
                    print(f"   ✅ Wave features extracted: {wave_features.shape[1]} features, {len(wave_features)} samples")
                    
                    # Test trading recommendations
                    print("   Testing trading recommendation generation...")
                    current_price = self.test_data['close'].iloc[-1]
                    recommendations = analyzer.generate_trading_recommendations(
                        analysis_results, current_price
                    )
                    
                    if recommendations and 'overall_recommendation' in recommendations:
                        rec = recommendations['overall_recommendation']
                        print(f"   ✅ Trading recommendations generated:")
                        print(f"      Action: {rec.get('action', 'N/A')}")
                        print(f"      Confidence: {rec.get('confidence', 'N/A'):.2f}")
                        print(f"      Position Size: {rec.get('position_size', 'N/A')}")
                        
                        self.test_results['advanced_analyzer'] = {
                            'status': 'success',
                            'timeframes_analyzed': len(analysis_results),
                            'features_extracted': wave_features.shape[1],
                            'recommendation_generated': True,
                            'recommendation': rec
                        }
                        return True
                    else:
                        print("   ⚠️ Trading recommendation generation failed")
                        self.test_results['advanced_analyzer'] = {
                            'status': 'partial_success',
                            'issue': 'recommendation_generation_failed'
                        }
                        return False
                else:
                    print("   ⚠️ Wave feature extraction failed")
                    self.test_results['advanced_analyzer'] = {
                        'status': 'partial_success',
                        'issue': 'feature_extraction_failed'
                    }
                    return False
            else:
                print("   ❌ Multi-timeframe analysis failed")
                self.test_results['advanced_analyzer'] = {
                    'status': 'failed',
                    'issue': 'analysis_failed'
                }
                return False
                
        except Exception as e:
            print(f"   ❌ Advanced Elliott Wave Analyzer test failed: {e}")
            self.test_results['advanced_analyzer'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_enhanced_dqn_agent(self):
        """Test Enhanced DQN Agent independently"""
        print("\n🤖 Testing Enhanced DQN Agent...")
        
        if not ENHANCED_DQN_AVAILABLE:
            print("⚠️ Enhanced DQN Agent not available - skipping test")
            self.test_results['enhanced_dqn'] = {'status': 'skipped', 'reason': 'not_available'}
            return False
        
        try:
            # Initialize Elliott Wave analyzer for DQN integration
            if ADVANCED_ANALYZER_AVAILABLE:
                analyzer = AdvancedElliottWaveAnalyzer(
                    timeframes=['1m', '5m'],
                    logger=None
                )
            else:
                analyzer = None
                print("   ⚠️ No Elliott Wave analyzer available - testing without integration")
            
            # Initialize Enhanced DQN Agent
            config = {
                'state_size': 20,
                'action_size': 6,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 1000,
                'batch_size': 16,
                'target_update': 50,
                'curriculum_learning': True,
                'elliott_wave_integration': analyzer is not None
            }
            
            enhanced_dqn = EnhancedDQNAgent(
                config=config,
                elliott_wave_analyzer=analyzer,
                logger=None
            )
            
            # Prepare simple test data for DQN
            print("   Testing DQN initialization and basic functionality...")
            
            # Create simple market features
            features = ['close', 'volume', 'high', 'low']
            market_data = self.test_data[features].copy()
            
            # Add some technical indicators
            market_data['sma_5'] = market_data['close'].rolling(5).mean()
            market_data['rsi'] = self._calculate_rsi(market_data['close'])
            market_data = market_data.dropna()
            
            if len(market_data) < 100:
                print("   ⚠️ Insufficient data for DQN testing")
                self.test_results['enhanced_dqn'] = {
                    'status': 'skipped',
                    'reason': 'insufficient_data'
                }
                return False
            
            # Test basic DQN functionality
            print("   Testing basic DQN training...")
            
            # Use smaller subset for faster testing
            test_subset = market_data.iloc[:100]
            price_data = test_subset[['close']].copy()
            
            if hasattr(enhanced_dqn, 'train_with_curriculum_learning'):
                print("   Testing curriculum learning...")
                results = enhanced_dqn.train_with_curriculum_learning(
                    market_data=test_subset,
                    price_data=price_data,
                    episodes=20,  # Reduced for testing
                    curriculum_stages=2
                )
                
                if results and 'curriculum_results' in results:
                    print("   ✅ Curriculum learning completed successfully")
                    self.test_results['enhanced_dqn'] = {
                        'status': 'success',
                        'curriculum_learning': True,
                        'episodes_completed': 20,
                        'curriculum_stages': 2
                    }
                    return True
                else:
                    print("   ⚠️ Curriculum learning completed but no detailed results")
                    self.test_results['enhanced_dqn'] = {
                        'status': 'partial_success',
                        'curriculum_learning': True,
                        'issue': 'no_detailed_results'
                    }
                    return True
            else:
                print("   Testing standard DQN training...")
                results = enhanced_dqn.train_agent(test_subset, episodes=10)
                
                if results:
                    print("   ✅ Standard DQN training completed")
                    self.test_results['enhanced_dqn'] = {
                        'status': 'success',
                        'curriculum_learning': False,
                        'episodes_completed': 10
                    }
                    return True
                else:
                    print("   ⚠️ DQN training completed but no results returned")
                    self.test_results['enhanced_dqn'] = {
                        'status': 'partial_success',
                        'issue': 'no_results_returned'
                    }
                    return False
                
        except Exception as e:
            print(f"   ❌ Enhanced DQN Agent test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.test_results['enhanced_dqn'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_enhanced_menu_integration(self):
        """Test the full Enhanced Menu integration"""
        print("\n🚀 Testing Enhanced Menu Integration...")
        
        if not ENHANCED_MENU_AVAILABLE:
            print("⚠️ Enhanced Menu not available - skipping test")
            self.test_results['enhanced_menu'] = {'status': 'skipped', 'reason': 'not_available'}
            return False
        
        try:
            # Configuration for testing
            config = {
                'elliott_wave': {
                    'timeframes': ['1m', '5m'],  # Reduced for testing
                    'target_auc': 0.70,
                    'max_features': 15  # Reduced for testing
                },
                'dqn': {
                    'state_size': 20,  # Reduced for testing
                    'learning_rate': 0.001,
                    'gamma': 0.95,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.01,
                    'epsilon_decay': 0.995,
                    'memory_size': 1000,  # Reduced for testing
                    'batch_size': 16,  # Reduced for testing
                    'target_update': 50
                }
            }
            
            # Create temporary data files for testing
            self._create_temp_data_files()
            
            print("   Initializing Enhanced Menu...")
            enhanced_menu = EnhancedMenu1ElliottWave(config=config)
            
            print("   Testing enhanced pipeline execution...")
            # Note: This might take a while due to full pipeline execution
            results = enhanced_menu.run()
            
            if results and results.get('success'):
                print("   ✅ Enhanced Menu integration test completed successfully")
                
                # Analyze results
                execution_status = results.get('execution_status', 'unknown')
                
                analysis_info = {}
                if 'elliott_wave_analysis' in results:
                    ew_analysis = results['elliott_wave_analysis']
                    analysis_info['elliott_wave_status'] = ew_analysis.get('analysis_status', 'unknown')
                    analysis_info['wave_features_count'] = ew_analysis.get('wave_features_count', 0)
                
                if 'dqn_training' in results:
                    dqn_results = results['dqn_training']
                    analysis_info['dqn_training_success'] = 'error' not in dqn_results
                    if 'curriculum_results' in dqn_results:
                        analysis_info['curriculum_learning'] = True
                
                if 'trading_recommendations' in results:
                    rec_results = results['trading_recommendations']
                    analysis_info['recommendations_generated'] = 'error' not in rec_results
                
                self.test_results['enhanced_menu'] = {
                    'status': 'success',
                    'execution_status': execution_status,
                    'analysis_info': analysis_info,
                    'full_results_available': True
                }
                return True
                
            else:
                print(f"   ⚠️ Enhanced Menu integration test completed with issues")
                execution_status = results.get('execution_status', 'unknown') if results else 'no_results'
                error_message = results.get('error_message', 'unknown') if results else 'no_results'
                
                self.test_results['enhanced_menu'] = {
                    'status': 'partial_success',
                    'execution_status': execution_status,
                    'error_message': error_message
                }
                return False
                
        except Exception as e:
            print(f"   ❌ Enhanced Menu integration test failed: {e}")
            self.test_results['enhanced_menu'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
        finally:
            # Clean up temporary files
            self._cleanup_temp_data_files()
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI for test data"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_temp_data_files(self):
        """Create temporary data files for testing"""
        try:
            # Create datacsv directory if it doesn't exist
            datacsv_dir = Path("datacsv")
            datacsv_dir.mkdir(exist_ok=True)
            
            # Save test data to CSV
            temp_file = datacsv_dir / "test_data.csv"
            self.test_data.reset_index().to_csv(temp_file, index=False)
            print(f"   Created temporary data file: {temp_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not create temporary data files: {e}")
    
    def _cleanup_temp_data_files(self):
        """Clean up temporary data files"""
        try:
            temp_file = Path("datacsv/test_data.csv")
            if temp_file.exists():
                temp_file.unlink()
                print(f"   Cleaned up temporary data file: {temp_file}")
        except Exception as e:
            print(f"   ⚠️ Could not clean up temporary files: {e}")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 ENHANCED ELLIOTT WAVE SYSTEM INTEGRATION TEST")
        print("="*60)
        
        # Generate test data
        self.generate_test_data(500)  # Reduced for faster testing
        
        # Run individual component tests
        test1_success = self.test_advanced_elliott_wave_analyzer()
        test2_success = self.test_enhanced_dqn_agent()
        
        # Run full integration test (optional, can be time-consuming)
        print("\n❓ Run full integration test? (This may take several minutes)")
        print("   Press Enter to skip, or type 'yes' to run full test...")
        user_input = input().strip().lower()
        
        test3_success = False
        if user_input == 'yes':
            test3_success = self.test_enhanced_menu_integration()
        else:
            print("⏭️ Skipping full integration test")
            self.test_results['enhanced_menu'] = {'status': 'skipped', 'reason': 'user_choice'}
        
        # Generate test report
        self._generate_test_report(test1_success, test2_success, test3_success)
        
        return self.test_results
    
    def _generate_test_report(self, test1_success, test2_success, test3_success):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📋 ENHANCED ELLIOTT WAVE SYSTEM TEST REPORT")
        print("="*60)
        
        total_time = time.time() - self.start_time
        
        print(f"⏱️ Total test duration: {total_time:.2f} seconds")
        print(f"📊 Test data: {len(self.test_data)} samples generated")
        
        print(f"\n🧪 Test Results:")
        
        # Advanced Elliott Wave Analyzer
        analyzer_result = self.test_results.get('advanced_analyzer', {})
        status = analyzer_result.get('status', 'unknown')
        if status == 'success':
            print(f"   ✅ Advanced Elliott Wave Analyzer: SUCCESS")
            print(f"      • Timeframes analyzed: {analyzer_result.get('timeframes_analyzed', 'N/A')}")
            print(f"      • Features extracted: {analyzer_result.get('features_extracted', 'N/A')}")
            print(f"      • Recommendations: {'YES' if analyzer_result.get('recommendation_generated') else 'NO'}")
        elif status == 'skipped':
            print(f"   ⏭️ Advanced Elliott Wave Analyzer: SKIPPED ({analyzer_result.get('reason', 'unknown')})")
        elif status == 'partial_success':
            print(f"   ⚠️ Advanced Elliott Wave Analyzer: PARTIAL SUCCESS ({analyzer_result.get('issue', 'unknown')})")
        else:
            print(f"   ❌ Advanced Elliott Wave Analyzer: FAILED ({analyzer_result.get('error', 'unknown')})")
        
        # Enhanced DQN Agent
        dqn_result = self.test_results.get('enhanced_dqn', {})
        status = dqn_result.get('status', 'unknown')
        if status == 'success':
            print(f"   ✅ Enhanced DQN Agent: SUCCESS")
            print(f"      • Curriculum learning: {'YES' if dqn_result.get('curriculum_learning') else 'NO'}")
            print(f"      • Episodes completed: {dqn_result.get('episodes_completed', 'N/A')}")
        elif status == 'skipped':
            print(f"   ⏭️ Enhanced DQN Agent: SKIPPED ({dqn_result.get('reason', 'unknown')})")
        elif status == 'partial_success':
            print(f"   ⚠️ Enhanced DQN Agent: PARTIAL SUCCESS ({dqn_result.get('issue', 'unknown')})")
        else:
            print(f"   ❌ Enhanced DQN Agent: FAILED ({dqn_result.get('error', 'unknown')})")
        
        # Enhanced Menu Integration
        menu_result = self.test_results.get('enhanced_menu', {})
        status = menu_result.get('status', 'unknown')
        if status == 'success':
            print(f"   ✅ Enhanced Menu Integration: SUCCESS")
            analysis_info = menu_result.get('analysis_info', {})
            print(f"      • Elliott Wave analysis: {analysis_info.get('elliott_wave_status', 'N/A')}")
            print(f"      • DQN training: {'SUCCESS' if analysis_info.get('dqn_training_success') else 'FAILED'}")
            print(f"      • Recommendations: {'YES' if analysis_info.get('recommendations_generated') else 'NO'}")
        elif status == 'skipped':
            print(f"   ⏭️ Enhanced Menu Integration: SKIPPED ({menu_result.get('reason', 'unknown')})")
        elif status == 'partial_success':
            print(f"   ⚠️ Enhanced Menu Integration: PARTIAL SUCCESS")
            print(f"      • Execution status: {menu_result.get('execution_status', 'unknown')}")
        else:
            print(f"   ❌ Enhanced Menu Integration: FAILED ({menu_result.get('error', 'unknown')})")
        
        # Overall assessment
        successful_tests = sum([test1_success, test2_success, test3_success])
        total_tests = 3
        
        print(f"\n📈 Overall Results: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Enhanced Elliott Wave System is fully functional!")
        elif successful_tests >= 2:
            print("✅ MOST TESTS PASSED - Enhanced Elliott Wave System is mostly functional")
        elif successful_tests >= 1:
            print("⚠️ SOME TESTS PASSED - Enhanced Elliott Wave System has partial functionality")
        else:
            print("❌ TESTS FAILED - Enhanced Elliott Wave System needs attention")
        
        print("="*60 + "\n")


def main():
    """Main function to run Enhanced Elliott Wave System integration tests"""
    print("🧪 Enhanced Elliott Wave System Integration Test Suite")
    print("This will test the new advanced modules and their integration\n")
    
    # Create and run test suite
    test_suite = EnhancedElliottWaveIntegrationTest()
    results = test_suite.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()
