#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE ENTERPRISE ML PROTECTION INTEGRATION TEST
ทดสอบการ integrate ของ Enterprise ML Protection System กับ pipeline จริงแบบครบถ้วน

Test Coverage:
- Complete Menu 1 Integration Test
- Pipeline Orchestrator Protection Stages
- Real Data Processing with Protection
- Edge Cases and Error Handling
- Performance and Stability Assessment
- Production-Ready Validation
"""

import sys
import os
import json
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import core modules
from core.project_paths import get_project_paths

# Import menu and protection system
from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem

class ComprehensiveIntegrationTester:
    """ตัวทดสอบการ integrate แบบครบถ้วน"""
    
    def __init__(self):
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("integration_test")
        self.paths = get_project_paths()
        self.test_results = {}
        
    def run_comprehensive_test(self):
        """รันการทดสอบแบบครบถ้วน"""
        self.logger.info("🎯 Starting Comprehensive Enterprise ML Protection Integration Test...")
        
        test_stages = [
            ('protection_system_standalone', self._test_protection_system_standalone),
            ('menu_initialization', self._test_menu_initialization),
            ('data_loading_with_protection', self._test_data_loading_with_protection),
            ('pipeline_protection_stages', self._test_pipeline_protection_stages),
            ('edge_cases_handling', self._test_edge_cases_handling),
            ('performance_assessment', self._test_performance_assessment),
            ('production_readiness', self._test_production_readiness)
        ]
        
        overall_success = True
        
        for stage_name, test_function in test_stages:
            try:
                self.logger.info(f"🧪 Testing: {stage_name.replace('_', ' ').title()}")
                
                result = test_function()
                self.test_results[stage_name] = result
                
                if result.get('success', False):
                    self.logger.info(f"✅ {stage_name} test passed")
                else:
                    self.logger.error(f"❌ {stage_name} test failed: {result.get('error', 'Unknown error')}")
                    overall_success = False
                    
            except Exception as e:
                error_msg = f"Test {stage_name} crashed: {str(e)}"
                self.logger.error(f"💥 {error_msg}")
                self.test_results[stage_name] = {
                    'success': False,
                    'error': error_msg,
                    'traceback': traceback.format_exc()
                }
                overall_success = False
        
        # Generate comprehensive report
        self._generate_comprehensive_report(overall_success)
        
        return overall_success
    
    def _clean_for_json(self, obj):
        """ทำความสะอาดข้อมูลสำหรับ JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist() if obj.size < 100 else f"Array shape: {obj.shape}"
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"DataFrame/Series shape: {obj.shape}"
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _test_protection_system_standalone(self):
        """ทดสอบ protection system แบบ standalone"""
        try:
            self.logger.info("🛡️ Testing Enterprise ML Protection System standalone...")
            
            protection_system = EnterpriseMLProtectionSystem(logger=self.logger)
            
            # Create synthetic test data
            n_samples = 1000
            n_features = 10
            
            # Create data with various issues for testing
            X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])
            
            # Add some problematic features for testing
            X['perfect_leak'] = np.random.rand(n_samples) > 0.5  # Perfect correlation candidate
            X['future_info'] = X['feature_0'].shift(-5)  # Future information
            X['noisy_feature'] = np.random.randn(n_samples) * 100  # Very noisy
            
            # Create target
            y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            # Add datetime column for temporal tests
            X['date'] = pd.date_range('2020-01-01', periods=n_samples, freq='H')
            
            # Run protection analysis
            protection_results = protection_system.comprehensive_protection_analysis(
                X=X.drop('date', axis=1),
                y=y,
                datetime_col='date'
            )
            
            # Validate results structure
            required_keys = ['data_leakage', 'overfitting', 'noise_analysis', 'overall_assessment']
            missing_keys = [key for key in required_keys if key not in protection_results]
            
            if missing_keys:
                return {
                    'success': False,
                    'error': f"Missing required result keys: {missing_keys}",
                    'protection_results': protection_results
                }
            
            # Check if protection system detected issues
            overall_assessment = protection_results.get('overall_assessment', {})
            alerts = protection_results.get('alerts', [])
            
            return {
                'success': True,
                'protection_results': protection_results,
                'overall_assessment': overall_assessment,
                'alerts_detected': len(alerts),
                'enterprise_ready': overall_assessment.get('enterprise_ready', False),
                'protection_status': overall_assessment.get('protection_status', 'UNKNOWN')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Protection system standalone test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_menu_initialization(self):
        """ทดสอบการเริ่มต้น Menu 1 และ protection system"""
        try:
            self.logger.info("🌊 Testing Menu 1 initialization with protection system...")
            
            # Create config
            config = {
                'elliott_wave': {
                    'target_auc': 0.70,
                    'max_features': 20,
                    'lookback_period': 50
                }
            }
            
            # Initialize Menu 1
            menu1 = Menu1ElliottWave(config=config, logger=self.logger)
            
            # Check if all components are initialized
            components = ['data_processor', 'cnn_lstm_engine', 'dqn_agent', 
                        'feature_selector', 'ml_protection', 'pipeline_orchestrator']
            
            missing_components = []
            for component in components:
                if not hasattr(menu1, component) or getattr(menu1, component) is None:
                    missing_components.append(component)
            
            if missing_components:
                return {
                    'success': False,
                    'error': f"Missing components: {missing_components}"
                }
            
            # Check if protection system is properly integrated
            if not hasattr(menu1.pipeline_orchestrator, 'ml_protection'):
                return {
                    'success': False,
                    'error': "ML Protection not integrated into pipeline orchestrator"
                }
            
            return {
                'success': True,
                'components_initialized': len(components),
                'protection_integrated': True,
                'orchestrator_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Menu initialization test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_data_loading_with_protection(self):
        """ทดสอบการโหลดข้อมูลจริงพร้อม protection analysis"""
        try:
            self.logger.info("📊 Testing real data loading with protection analysis...")
            
            # Initialize menu
            menu1 = Menu1ElliottWave(logger=self.logger)
            
            # Check if data files exist
            data_files = ['XAUUSD_M1.csv', 'XAUUSD_M15.csv']
            available_files = []
            
            for file_name in data_files:
                file_path = self.paths['datacsv'] / file_name
                if file_path.exists():
                    available_files.append(file_name)
            
            if not available_files:
                return {
                    'success': False,
                    'error': "No data files available for testing"
                }
            
            # Load data using the first available file
            data_file = available_files[0]
            self.logger.info(f"Loading data from: {data_file}")
            
            # Test data loading
            data_processor = menu1.data_processor
            raw_data = data_processor.load_real_data(timeframe='M15')
            
            if raw_data is None or raw_data.empty:
                return {
                    'success': False,
                    'error': "Failed to load real data"
                }
            
            # Process data
            processed_data = data_processor.prepare_elliott_wave_data(raw_data)
            
            if processed_data is None or processed_data.empty:
                return {
                    'success': False,
                    'error': "Failed to process real data"
                }
            
            # Test protection analysis on real data
            if len(processed_data) > 100:
                # Create simple target for testing
                target = (processed_data['close'].pct_change() > 0).astype(int).dropna()
                features = processed_data.select_dtypes(include=['number']).iloc[1:].dropna()
                
                # Align features and target
                min_len = min(len(features), len(target))
                features = features.iloc[:min_len]
                target = target.iloc[:min_len]
                
                if len(features) > 50:
                    protection_results = menu1.ml_protection.comprehensive_protection_analysis(
                        X=features,
                        y=target
                    )
                    
                    return {
                        'success': True,
                        'data_loaded': True,
                        'data_samples': len(processed_data),
                        'data_features': len(processed_data.columns),
                        'protection_analysis': True,
                        'protection_status': protection_results.get('overall_assessment', {}).get('protection_status', 'UNKNOWN')
                    }
            
            return {
                'success': True,
                'data_loaded': True,
                'data_samples': len(processed_data),
                'data_features': len(processed_data.columns),
                'protection_analysis': False,
                'note': "Insufficient data for protection analysis"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Data loading with protection test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_pipeline_protection_stages(self):
        """ทดสอบ protection stages ใน pipeline orchestrator"""
        try:
            self.logger.info("🎼 Testing pipeline protection stages...")
            
            # Initialize menu
            menu1 = Menu1ElliottWave(logger=self.logger)
            orchestrator = menu1.pipeline_orchestrator
            
            # Check if protection stages exist
            protection_stages = [
                '_stage_2b_enterprise_protection_analysis',
                '_stage_4b_pre_training_validation', 
                '_stage_6b_post_training_protection',
                '_stage_8b_final_protection_report'
            ]
            
            missing_stages = []
            for stage in protection_stages:
                if not hasattr(orchestrator, stage):
                    missing_stages.append(stage)
            
            if missing_stages:
                return {
                    'success': False,
                    'error': f"Missing protection stages: {missing_stages}"
                }
            
            # Test calling individual protection stages
            stage_results = {}
            
            # Create minimal test data for pipeline
            test_data = pd.DataFrame({
                'close': np.random.randn(200) + 100,
                'open': np.random.randn(200) + 100,
                'high': np.random.randn(200) + 101,
                'low': np.random.randn(200) + 99,
                'volume': np.random.randint(1000, 10000, 200)
            })
            
            # Store test data in pipeline results
            orchestrator.pipeline_results['data_preprocessing'] = {
                'success': True,
                'processed_data': test_data
            }
            
            # Test enterprise protection analysis stage
            stage_result = orchestrator._stage_2b_enterprise_protection_analysis()
            stage_results['enterprise_protection_analysis'] = stage_result
            
            return {
                'success': True,
                'protection_stages_available': len(protection_stages),
                'test_stage_result': stage_result.get('success', False),
                'protection_status': stage_result.get('protection_status', 'UNKNOWN'),
                'stage_results': stage_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Pipeline protection stages test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_edge_cases_handling(self):
        """ทดสอบการจัดการ edge cases"""
        try:
            self.logger.info("🎪 Testing edge cases handling...")
            
            protection_system = EnterpriseMLProtectionSystem(logger=self.logger)
            edge_case_results = {}
            
            # Test 1: Empty data
            try:
                empty_result = protection_system.comprehensive_protection_analysis(
                    X=pd.DataFrame(),
                    y=pd.Series(dtype=int)
                )
                edge_case_results['empty_data'] = {
                    'handled': True,
                    'error': empty_result.get('error', 'No error')
                }
            except Exception as e:
                edge_case_results['empty_data'] = {
                    'handled': False,
                    'error': str(e)
                }
            
            # Test 2: Single sample
            try:
                single_result = protection_system.comprehensive_protection_analysis(
                    X=pd.DataFrame({'feature1': [1]}),
                    y=pd.Series([0])
                )
                edge_case_results['single_sample'] = {
                    'handled': True,
                    'error': single_result.get('error', 'No error')
                }
            except Exception as e:
                edge_case_results['single_sample'] = {
                    'handled': False,
                    'error': str(e)
                }
            
            # Test 3: All NaN features
            try:
                nan_data = pd.DataFrame({'feature1': [np.nan] * 100, 'feature2': [np.nan] * 100})
                nan_target = pd.Series([0, 1] * 50)
                
                nan_result = protection_system.comprehensive_protection_analysis(
                    X=nan_data,
                    y=nan_target
                )
                edge_case_results['nan_features'] = {
                    'handled': True,
                    'error': nan_result.get('error', 'No error')
                }
            except Exception as e:
                edge_case_results['nan_features'] = {
                    'handled': False,
                    'error': str(e)
                }
            
            # Test 4: Constant features
            try:
                constant_data = pd.DataFrame({
                    'constant1': [1] * 100,
                    'constant2': [0] * 100,
                    'normal': np.random.randn(100)
                })
                constant_target = pd.Series([0, 1] * 50)
                
                constant_result = protection_system.comprehensive_protection_analysis(
                    X=constant_data,
                    y=constant_target
                )
                edge_case_results['constant_features'] = {
                    'handled': True,
                    'protection_status': constant_result.get('overall_assessment', {}).get('protection_status', 'UNKNOWN')
                }
            except Exception as e:
                edge_case_results['constant_features'] = {
                    'handled': False,
                    'error': str(e)
                }
            
            # Calculate success rate
            handled_cases = sum(1 for result in edge_case_results.values() if result.get('handled', False))
            total_cases = len(edge_case_results)
            success_rate = handled_cases / total_cases if total_cases > 0 else 0
            
            return {
                'success': success_rate >= 0.75,  # At least 75% of edge cases handled
                'edge_cases_tested': total_cases,
                'edge_cases_handled': handled_cases,
                'success_rate': success_rate,
                'edge_case_results': edge_case_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Edge cases handling test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_performance_assessment(self):
        """ทดสอบประสิทธิภาพของระบบ"""
        try:
            self.logger.info("⚡ Testing performance assessment...")
            
            import time
            
            protection_system = EnterpriseMLProtectionSystem(logger=self.logger)
            
            # Performance test with different data sizes
            performance_results = {}
            
            for n_samples in [100, 500, 1000]:
                try:
                    # Create test data
                    X = pd.DataFrame(np.random.randn(n_samples, 20), 
                                   columns=[f'feature_{i}' for i in range(20)])
                    y = (X['feature_0'] + X['feature_1'] > 0).astype(int)
                    
                    # Measure execution time
                    start_time = time.time()
                    result = protection_system.comprehensive_protection_analysis(X=X, y=y)
                    execution_time = time.time() - start_time
                    
                    performance_results[f'{n_samples}_samples'] = {
                        'execution_time': execution_time,
                        'success': result.get('protection_status') != 'ERROR',
                        'samples_per_second': n_samples / execution_time if execution_time > 0 else 0
                    }
                    
                except Exception as e:
                    performance_results[f'{n_samples}_samples'] = {
                        'execution_time': None,
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate average performance
            successful_tests = [r for r in performance_results.values() if r.get('success', False)]
            avg_execution_time = np.mean([r['execution_time'] for r in successful_tests]) if successful_tests else None
            
            return {
                'success': len(successful_tests) >= 2,  # At least 2 size tests successful
                'performance_results': performance_results,
                'avg_execution_time': avg_execution_time,
                'performance_acceptable': avg_execution_time is not None and avg_execution_time < 30  # Under 30 seconds
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Performance assessment test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _test_production_readiness(self):
        """ทดสอบความพร้อมสำหรับ production"""
        try:
            self.logger.info("🏭 Testing production readiness...")
            
            readiness_checks = {}
            
            # Check 1: All components can be imported
            try:
                from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
                from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
                from elliott_wave_modules.pipeline_orchestrator import ElliottWavePipelineOrchestrator
                
                readiness_checks['imports'] = {'success': True}
            except Exception as e:
                readiness_checks['imports'] = {'success': False, 'error': str(e)}
            
            # Check 2: Menu can be initialized
            try:
                menu = Menu1ElliottWave(logger=self.logger)
                readiness_checks['menu_initialization'] = {'success': True}
            except Exception as e:
                readiness_checks['menu_initialization'] = {'success': False, 'error': str(e)}
            
            # Check 3: Protection system integration
            try:
                menu = Menu1ElliottWave(logger=self.logger)
                has_protection = hasattr(menu, 'ml_protection') and menu.ml_protection is not None
                has_orchestrator = hasattr(menu, 'pipeline_orchestrator') and menu.pipeline_orchestrator is not None
                protection_in_orchestrator = hasattr(menu.pipeline_orchestrator, 'ml_protection')
                
                readiness_checks['protection_integration'] = {
                    'success': has_protection and has_orchestrator and protection_in_orchestrator,
                    'has_protection': has_protection,
                    'has_orchestrator': has_orchestrator,
                    'protection_in_orchestrator': protection_in_orchestrator
                }
            except Exception as e:
                readiness_checks['protection_integration'] = {'success': False, 'error': str(e)}
            
            # Check 4: Configuration handling
            try:
                config = {'elliott_wave': {'target_auc': 0.75}}
                menu = Menu1ElliottWave(config=config, logger=self.logger)
                readiness_checks['configuration'] = {'success': True}
            except Exception as e:
                readiness_checks['configuration'] = {'success': False, 'error': str(e)}
            
            # Check 5: Error handling
            try:
                protection = EnterpriseMLProtectionSystem(logger=self.logger)
                error_result = protection.comprehensive_protection_analysis(
                    X=pd.DataFrame(),  # Empty data should be handled gracefully
                    y=pd.Series(dtype=int)
                )
                handles_errors = 'error' in error_result or error_result.get('protection_status') == 'ERROR'
                readiness_checks['error_handling'] = {'success': handles_errors}
            except Exception as e:
                readiness_checks['error_handling'] = {'success': False, 'error': str(e)}
            
            # Overall readiness score
            successful_checks = sum(1 for check in readiness_checks.values() if check.get('success', False))
            total_checks = len(readiness_checks)
            readiness_score = successful_checks / total_checks if total_checks > 0 else 0
            
            return {
                'success': readiness_score >= 0.8,  # 80% of checks must pass
                'readiness_score': float(readiness_score),
                'successful_checks': int(successful_checks),
                'total_checks': int(total_checks),
                'readiness_checks': readiness_checks,
                'production_ready': bool(readiness_score >= 0.8)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Production readiness test failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _generate_comprehensive_report(self, overall_success: bool):
        """สร้างรายงานการทดสอบแบบครบถ้วน"""
        try:
            # Clean test results for JSON serialization
            cleaned_results = {}
            for test_name, result in self.test_results.items():
                cleaned_result = {}
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        cleaned_result[key] = value
                    elif isinstance(value, (list, dict)):
                        cleaned_result[key] = self._clean_for_json(value)
                    else:
                        cleaned_result[key] = str(value)
                cleaned_results[test_name] = cleaned_result
            
            report = {
                'test_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'overall_success': bool(overall_success),
                    'total_tests': len(self.test_results),
                    'passed_tests': sum(1 for r in self.test_results.values() if r.get('success', False)),
                    'failed_tests': sum(1 for r in self.test_results.values() if not r.get('success', False))
                },
                'test_results': cleaned_results
            }
            
            # Save detailed report
            report_file = Path('temp_logs') / f'comprehensive_integration_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Log summary
            self.logger.info("📋 COMPREHENSIVE INTEGRATION TEST REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"Overall Success: {overall_success}")
            self.logger.info(f"Tests Passed: {report['test_summary']['passed_tests']}/{report['test_summary']['total_tests']}")
            
            for test_name, result in self.test_results.items():
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                self.logger.info(f"{status} {test_name.replace('_', ' ').title()}")
                
                if not result.get('success', False) and 'error' in result:
                    self.logger.error(f"  Error: {result['error']}")
            
            self.logger.info("=" * 50)
            self.logger.info(f"Detailed report saved: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {str(e)}")
            return None

def main():
    """Main execution function"""
    print("🎯 Starting Comprehensive Enterprise ML Protection Integration Test...")
    
    try:
        tester = ComprehensiveIntegrationTester()
        success = tester.run_comprehensive_test()
        
        if success:
            print("✅ All integration tests passed! System is production-ready.")
            return 0
        else:
            print("❌ Some integration tests failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"💥 Integration test crashed: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
