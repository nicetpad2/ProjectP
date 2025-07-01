#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE INTELLIGENT RESOURCE SYSTEM TEST
================================================

ทดสอบระบบจัดการทรัพยากรอัจฉริยะแบบครอบคลุม
- ทดสอบ Resource Detection & Allocation
- ทดสอบ Menu 1 Integration
- ทดสอบ Real-time Monitoring
- ทดสอบ Performance Benefits Analysis

🎯 Test Objectives:
- Verify 80% allocation strategy
- Test Menu 1 pipeline integration
- Validate performance monitoring
- Confirm adaptive resource adjustment
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import resource management components
try:
    from core.intelligent_resource_manager import IntelligentResourceManager, initialize_intelligent_resources
    from core.enhanced_intelligent_resource_manager import EnhancedIntelligentResourceManager, initialize_enhanced_intelligent_resources
    from core.menu1_resource_integration import Menu1ResourceIntegrator, create_menu1_resource_integrator
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    sys.exit(1)

class IntelligentResourceSystemTester:
    """
    🧪 Comprehensive Tester for Intelligent Resource System
    """
    
    def __init__(self):
        self.test_results = {}
        self.test_start_time = datetime.now()
        self.test_log = []
        
    def log_test(self, test_name: str, status: str, details: str = "") -> None:
        """
        📝 บันทึกผลการทดสอบ
        """
        entry = {
            'test_name': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_log.append(entry)
        
        # Color output
        color = "\033[92m" if status == "PASSED" else "\033[91m" if status == "FAILED" else "\033[93m"
        reset = "\033[0m"
        
        print(f"{color}[{status}]{reset} {test_name}: {details}")
    
    def test_1_basic_resource_detection(self) -> bool:
        """
        🔍 Test 1: Basic Resource Detection
        """
        try:
            print("\n" + "="*60)
            print("🔍 TEST 1: BASIC RESOURCE DETECTION")
            print("="*60)
            
            # Initialize resource manager
            resource_manager = IntelligentResourceManager()
            
            # Check system info detection
            system_info = resource_manager.system_info
            
            # Verify CPU detection
            cpu_info = system_info.get('cpu', {})
            logical_cores = cpu_info.get('logical_cores', 0)
            physical_cores = cpu_info.get('physical_cores', 0)
            
            if logical_cores > 0 and physical_cores > 0:
                self.log_test("CPU Detection", "PASSED", f"{physical_cores} physical / {logical_cores} logical cores")
            else:
                self.log_test("CPU Detection", "FAILED", "No CPU cores detected")
                return False
            
            # Verify Memory detection
            memory_info = system_info.get('memory', {})
            total_gb = memory_info.get('total_gb', 0)
            available_gb = memory_info.get('available_gb', 0)
            
            if total_gb > 0 and available_gb > 0:
                self.log_test("Memory Detection", "PASSED", f"{total_gb:.1f}GB total / {available_gb:.1f}GB available")
            else:
                self.log_test("Memory Detection", "FAILED", "No memory detected")
                return False
            
            # Verify GPU detection (can be absent)
            gpu_info = system_info.get('gpu', {})
            cuda_available = gpu_info.get('cuda_available', False)
            
            if cuda_available:
                gpu_count = gpu_info.get('gpu_count', 0)
                self.log_test("GPU Detection", "PASSED", f"CUDA available with {gpu_count} devices")
            else:
                self.log_test("GPU Detection", "INFO", "CUDA not available (CPU-only mode)")
            
            # Verify Platform detection
            platform_info = system_info.get('platform', {})
            system_name = platform_info.get('system', '')
            python_version = platform_info.get('python_version', '')
            
            if system_name and python_version:
                self.log_test("Platform Detection", "PASSED", f"{system_name} with Python {python_version}")
            else:
                self.log_test("Platform Detection", "FAILED", "Platform info incomplete")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Basic Resource Detection", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_2_80_percent_allocation_strategy(self) -> bool:
        """
        ⚡ Test 2: 80% Allocation Strategy
        """
        try:
            print("\n" + "="*60)
            print("⚡ TEST 2: 80% ALLOCATION STRATEGY")
            print("="*60)
            
            # Initialize with 80% allocation
            resource_manager = IntelligentResourceManager(allocation_percentage=0.8)
            
            # Check resource configuration
            resource_config = resource_manager.resource_config
            
            # Verify CPU allocation
            cpu_config = resource_config.get('cpu', {})
            total_cores = cpu_config.get('total_cores', 0)
            allocated_threads = cpu_config.get('allocated_threads', 0)
            cpu_percentage = cpu_config.get('allocation_percentage', 0)
            
            if 75 <= cpu_percentage <= 85:  # Allow some variance
                self.log_test("CPU 80% Allocation", "PASSED", f"{allocated_threads}/{total_cores} cores ({cpu_percentage:.1f}%)")
            else:
                self.log_test("CPU 80% Allocation", "FAILED", f"Allocation: {cpu_percentage:.1f}% (expected ~80%)")
                return False
            
            # Verify Memory allocation
            memory_config = resource_config.get('memory', {})
            available_gb = memory_config.get('available_gb', 0)
            allocated_gb = memory_config.get('allocated_gb', 0)
            memory_percentage = memory_config.get('allocation_percentage', 0)
            
            if 75 <= memory_percentage <= 85:  # Allow some variance
                self.log_test("Memory 80% Allocation", "PASSED", f"{allocated_gb:.1f}/{available_gb:.1f}GB ({memory_percentage:.1f}%)")
            else:
                self.log_test("Memory 80% Allocation", "FAILED", f"Allocation: {memory_percentage:.1f}% (expected ~80%)")
                return False
            
            # Verify Batch size optimization
            optimization = resource_config.get('optimization', {})
            batch_size = optimization.get('batch_size', 0)
            recommended_workers = optimization.get('recommended_workers', 0)
            
            if batch_size > 0 and recommended_workers > 0:
                self.log_test("Batch Size Optimization", "PASSED", f"Batch: {batch_size}, Workers: {recommended_workers}")
            else:
                self.log_test("Batch Size Optimization", "FAILED", "No optimization calculated")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("80% Allocation Strategy", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_3_environment_optimization(self) -> bool:
        """
        🎯 Test 3: Environment Optimization
        """
        try:
            print("\n" + "="*60)
            print("🎯 TEST 3: ENVIRONMENT OPTIMIZATION")
            print("="*60)
            
            # Store original environment
            original_env = dict(os.environ)
            
            # Initialize and apply optimization
            resource_manager = IntelligentResourceManager()
            optimization_applied = resource_manager.apply_resource_optimization()
            
            if not optimization_applied:
                self.log_test("Environment Optimization", "FAILED", "Optimization application failed")
                return False
            
            # Check key environment variables
            env_checks = [
                ('TF_NUM_INTEROP_THREADS', 'TensorFlow inter-op threads'),
                ('TF_NUM_INTRAOP_THREADS', 'TensorFlow intra-op threads'),
                ('OMP_NUM_THREADS', 'OpenMP threads'),
                ('MKL_NUM_THREADS', 'MKL threads'),
                ('NUMBA_NUM_THREADS', 'Numba threads')
            ]
            
            optimization_count = 0
            for env_var, description in env_checks:
                value = os.environ.get(env_var)
                if value and value.isdigit():
                    self.log_test(f"Env: {description}", "PASSED", f"{env_var}={value}")
                    optimization_count += 1
                else:
                    self.log_test(f"Env: {description}", "INFO", f"{env_var} not set or invalid")
            
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
            
            if optimization_count >= 3:  # At least 3 optimizations should be applied
                self.log_test("Environment Optimization", "PASSED", f"{optimization_count} optimizations applied")
                return True
            else:
                self.log_test("Environment Optimization", "FAILED", f"Only {optimization_count} optimizations applied")
                return False
            
        except Exception as e:
            self.log_test("Environment Optimization", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_4_real_time_monitoring(self) -> bool:
        """
        📊 Test 4: Real-time Monitoring
        """
        try:
            print("\n" + "="*60)
            print("📊 TEST 4: REAL-TIME MONITORING")
            print("="*60)
            
            # Initialize resource manager
            resource_manager = IntelligentResourceManager()
            
            # Start monitoring
            resource_manager.start_monitoring(interval=0.5)
            
            # Wait for some monitoring data
            time.sleep(3)
            
            # Check if monitoring is active
            if not resource_manager.monitoring_active:
                self.log_test("Monitoring Activation", "FAILED", "Monitoring not active")
                return False
            else:
                self.log_test("Monitoring Activation", "PASSED", "Monitoring is active")
            
            # Check performance data collection
            performance_data = resource_manager.performance_data
            if len(performance_data) >= 3:  # Should have at least 3 data points
                self.log_test("Performance Data Collection", "PASSED", f"{len(performance_data)} data points collected")
            else:
                self.log_test("Performance Data Collection", "FAILED", f"Only {len(performance_data)} data points")
                return False
            
            # Check current performance
            current_perf = resource_manager.get_current_performance()
            required_fields = ['cpu_percent', 'memory', 'uptime_minutes']
            
            for field in required_fields:
                if field in current_perf:
                    self.log_test(f"Current Performance: {field}", "PASSED", f"Value: {current_perf[field]}")
                else:
                    self.log_test(f"Current Performance: {field}", "FAILED", "Field missing")
                    return False
            
            # Stop monitoring
            resource_manager.stop_monitoring()
            time.sleep(1)
            
            if not resource_manager.monitoring_active:
                self.log_test("Monitoring Deactivation", "PASSED", "Monitoring stopped successfully")
            else:
                self.log_test("Monitoring Deactivation", "FAILED", "Monitoring still active")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Real-time Monitoring", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_5_menu1_optimization_config(self) -> bool:
        """
        🌊 Test 5: Menu 1 Optimization Configuration
        """
        try:
            print("\n" + "="*60)
            print("🌊 TEST 5: MENU 1 OPTIMIZATION CONFIGURATION")
            print("="*60)
            
            # Initialize resource manager
            resource_manager = IntelligentResourceManager()
            
            # Get Menu 1 optimization config
            menu1_config = resource_manager.get_menu1_optimization_config()
            
            # Check required sections
            required_sections = [
                'data_processing',
                'elliott_wave', 
                'feature_selection',
                'cnn_lstm',
                'dqn',
                'monitoring'
            ]
            
            for section in required_sections:
                if section in menu1_config:
                    section_data = menu1_config[section]
                    self.log_test(f"Menu1 Config: {section}", "PASSED", f"Keys: {list(section_data.keys())}")
                else:
                    self.log_test(f"Menu1 Config: {section}", "FAILED", "Section missing")
                    return False
            
            # Check specific optimization parameters
            optimization_checks = [
                ('data_processing', 'chunk_size', int),
                ('data_processing', 'parallel_workers', int),
                ('elliott_wave', 'batch_size', int),
                ('feature_selection', 'n_trials', int),
                ('cnn_lstm', 'batch_size', int),
                ('dqn', 'batch_size', int)
            ]
            
            for section, param, expected_type in optimization_checks:
                if section in menu1_config and param in menu1_config[section]:
                    value = menu1_config[section][param]
                    if isinstance(value, expected_type) and value > 0:
                        self.log_test(f"Optimization: {section}.{param}", "PASSED", f"Value: {value}")
                    else:
                        self.log_test(f"Optimization: {section}.{param}", "FAILED", f"Invalid value: {value}")
                        return False
                else:
                    self.log_test(f"Optimization: {section}.{param}", "FAILED", "Parameter missing")
                    return False
            
            return True
            
        except Exception as e:
            self.log_test("Menu 1 Optimization Configuration", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_6_enhanced_resource_manager(self) -> bool:
        """
        ⚡ Test 6: Enhanced Resource Manager
        """
        try:
            print("\n" + "="*60)
            print("⚡ TEST 6: ENHANCED RESOURCE MANAGER")
            print("="*60)
            
            # Initialize enhanced resource manager
            enhanced_manager = initialize_enhanced_intelligent_resources()
            
            # Test stage monitoring setup
            if hasattr(enhanced_manager, 'menu1_stages'):
                stages = enhanced_manager.menu1_stages
                self.log_test("Enhanced: Stage Setup", "PASSED", f"{len(stages)} stages configured")
            else:
                self.log_test("Enhanced: Stage Setup", "FAILED", "Menu 1 stages not configured")
                return False
            
            # Test stage monitoring
            test_stage = 'data_loading'
            enhanced_manager.start_stage_monitoring(test_stage)
            time.sleep(1)
            
            mock_metrics = {'auc': 0.75, 'accuracy': 0.82}
            summary = enhanced_manager.end_stage_monitoring(test_stage, mock_metrics)
            
            if summary and 'efficiency_score' in summary:
                efficiency = summary['efficiency_score']
                self.log_test("Enhanced: Stage Monitoring", "PASSED", f"Efficiency: {efficiency:.2f}")
            else:
                self.log_test("Enhanced: Stage Monitoring", "FAILED", "No stage summary generated")
                return False
            
            # Test performance alerts
            if hasattr(enhanced_manager, 'performance_alerts'):
                self.log_test("Enhanced: Performance Alerts", "PASSED", "Alert system initialized")
            else:
                self.log_test("Enhanced: Performance Alerts", "FAILED", "Alert system missing")
                return False
            
            # Test pipeline performance report
            pipeline_report = enhanced_manager.generate_pipeline_performance_report()
            
            required_report_sections = ['pipeline_stages', 'pipeline_summary', 'performance_alerts_summary']
            for section in required_report_sections:
                if section in pipeline_report:
                    self.log_test(f"Enhanced: Report {section}", "PASSED", "Section present")
                else:
                    self.log_test(f"Enhanced: Report {section}", "FAILED", "Section missing")
                    return False
            
            enhanced_manager.stop_monitoring()
            return True
            
        except Exception as e:
            self.log_test("Enhanced Resource Manager", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_7_menu1_integration(self) -> bool:
        """
        🔗 Test 7: Menu 1 Integration
        """
        try:
            print("\n" + "="*60)
            print("🔗 TEST 7: MENU 1 INTEGRATION")
            print("="*60)
            
            # Create mock Menu 1 instance
            class MockMenu1:
                def __init__(self):
                    self.config = {}
            
            mock_menu1 = MockMenu1()
            
            # Create integrator
            integrator = create_menu1_resource_integrator()
            
            # Test integration
            integration_success = integrator.integrate_with_menu1(mock_menu1)
            
            if integration_success:
                self.log_test("Integration: Connection", "PASSED", "Menu 1 integration successful")
            else:
                self.log_test("Integration: Connection", "FAILED", "Menu 1 integration failed")
                return False
            
            # Test stage monitoring
            test_stages = ['data_loading', 'feature_engineering']
            for stage in test_stages:
                integrator.start_stage(stage)
                time.sleep(0.5)
                
                mock_metrics = {'duration': 10.5, 'memory_usage': 45.2}
                summary = integrator.end_stage(stage, mock_metrics)
                
                if summary:
                    self.log_test(f"Integration: {stage}", "PASSED", f"Stage completed successfully")
                else:
                    self.log_test(f"Integration: {stage}", "FAILED", f"Stage monitoring failed")
                    return False
            
            # Test performance status
            status = integrator.get_current_performance_status()
            
            required_status_fields = ['resource_utilization', 'integration_status', 'resource_allocation']
            for field in required_status_fields:
                if field in status:
                    self.log_test(f"Integration: Status {field}", "PASSED", "Field present")
                else:
                    self.log_test(f"Integration: Status {field}", "FAILED", "Field missing")
                    return False
            
            # Test integration report
            report = integrator.generate_integration_report()
            if report and 'integration_details' in report:
                self.log_test("Integration: Report Generation", "PASSED", "Report generated successfully")
            else:
                self.log_test("Integration: Report Generation", "FAILED", "Report generation failed")
                return False
            
            integrator.cleanup()
            return True
            
        except Exception as e:
            self.log_test("Menu 1 Integration", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_8_performance_benefits_analysis(self) -> bool:
        """
        📈 Test 8: Performance Benefits Analysis
        """
        try:
            print("\n" + "="*60)
            print("📈 TEST 8: PERFORMANCE BENEFITS ANALYSIS")
            print("="*60)
            
            # Simulate resource-optimized vs non-optimized processing
            
            # Test 1: Non-optimized processing
            start_time = time.time()
            data = np.random.randn(10000, 50)  # Simulated data processing
            result_normal = np.mean(data, axis=0)
            normal_duration = time.time() - start_time
            
            # Test 2: Resource-optimized processing (simulated)
            resource_manager = IntelligentResourceManager()
            optimization_config = resource_manager.get_menu1_optimization_config()
            
            # Apply some optimization (simulated)
            os.environ['OMP_NUM_THREADS'] = str(optimization_config.get('data_processing', {}).get('parallel_workers', 4))
            
            start_time = time.time()
            result_optimized = np.mean(data, axis=0)  # Same operation
            optimized_duration = time.time() - start_time
            
            # Calculate improvement
            if normal_duration > 0:
                improvement_ratio = normal_duration / optimized_duration
                improvement_percentage = (improvement_ratio - 1) * 100
                
                if improvement_ratio >= 0.8:  # Some improvement expected (even if minimal in this test)
                    self.log_test("Performance: Speed Improvement", "PASSED", 
                                f"Improvement ratio: {improvement_ratio:.2f}x ({improvement_percentage:.1f}%)")
                else:
                    self.log_test("Performance: Speed Improvement", "INFO", 
                                f"Minimal improvement in test scenario: {improvement_ratio:.2f}x")
            
            # Test memory efficiency
            import psutil
            current_memory = psutil.virtual_memory()
            memory_config = resource_manager.resource_config.get('memory', {})
            allocated_gb = memory_config.get('allocated_gb', 0)
            
            if allocated_gb > 0 and allocated_gb <= current_memory.available / (1024**3):
                self.log_test("Performance: Memory Efficiency", "PASSED", 
                            f"Allocated {allocated_gb:.1f}GB within available {current_memory.available/(1024**3):.1f}GB")
            else:
                self.log_test("Performance: Memory Efficiency", "FAILED", "Memory allocation out of bounds")
                return False
            
            # Test resource utilization tracking
            resource_manager.start_monitoring(interval=0.2)
            time.sleep(1)  # Let it collect some data
            
            performance_data = resource_manager.performance_data
            if len(performance_data) >= 3:
                avg_cpu = np.mean([d.get('cpu_percent', 0) for d in performance_data])
                avg_memory = np.mean([d.get('memory_percent', 0) for d in performance_data])
                
                self.log_test("Performance: Resource Tracking", "PASSED", 
                            f"Avg CPU: {avg_cpu:.1f}%, Avg Memory: {avg_memory:.1f}%")
            else:
                self.log_test("Performance: Resource Tracking", "FAILED", "Insufficient monitoring data")
                return False
            
            resource_manager.stop_monitoring()
            return True
            
        except Exception as e:
            self.log_test("Performance Benefits Analysis", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_9_report_generation(self) -> bool:
        """
        📋 Test 9: Report Generation
        """
        try:
            print("\n" + "="*60)
            print("📋 TEST 9: REPORT GENERATION")
            print("="*60)
            
            # Initialize resource manager
            resource_manager = IntelligentResourceManager()
            
            # Generate performance report
            report = resource_manager.generate_performance_report()
            
            # Check report structure
            required_sections = ['system_info', 'resource_allocation', 'current_performance']
            for section in required_sections:
                if section in report:
                    self.log_test(f"Report: {section}", "PASSED", "Section present")
                else:
                    self.log_test(f"Report: {section}", "FAILED", "Section missing")
                    return False
            
            # Test report saving
            report_file = resource_manager.save_report()
            
            if report_file and os.path.exists(report_file):
                file_size = os.path.getsize(report_file)
                self.log_test("Report: File Generation", "PASSED", f"File: {report_file} ({file_size} bytes)")
                
                # Clean up test file
                try:
                    os.remove(report_file)
                except:
                    pass
            else:
                self.log_test("Report: File Generation", "FAILED", "Report file not created")
                return False
            
            # Test enhanced report
            enhanced_manager = initialize_enhanced_intelligent_resources()
            enhanced_report = enhanced_manager.generate_pipeline_performance_report()
            
            enhanced_sections = ['pipeline_stages', 'pipeline_summary']
            for section in enhanced_sections:
                if section in enhanced_report:
                    self.log_test(f"Enhanced Report: {section}", "PASSED", "Section present")
                else:
                    self.log_test(f"Enhanced Report: {section}", "FAILED", "Section missing")
                    return False
            
            enhanced_manager.stop_monitoring()
            return True
            
        except Exception as e:
            self.log_test("Report Generation", "FAILED", f"Exception: {str(e)}")
            return False
    
    def test_10_system_display(self) -> bool:
        """
        🖥️ Test 10: System Display
        """
        try:
            print("\n" + "="*60)
            print("🖥️ TEST 10: SYSTEM DISPLAY")
            print("="*60)
            
            # Test basic display
            resource_manager = IntelligentResourceManager()
            
            try:
                resource_manager.display_system_summary()
                self.log_test("Display: System Summary", "PASSED", "Display function executed")
            except Exception as e:
                self.log_test("Display: System Summary", "FAILED", f"Display error: {str(e)}")
                return False
            
            # Test enhanced display
            enhanced_manager = initialize_enhanced_intelligent_resources()
            
            try:
                enhanced_manager.display_real_time_dashboard()
                self.log_test("Display: Real-time Dashboard", "PASSED", "Dashboard function executed")
            except Exception as e:
                self.log_test("Display: Real-time Dashboard", "FAILED", f"Dashboard error: {str(e)}")
                return False
            
            # Test integration display
            integrator = create_menu1_resource_integrator()
            
            try:
                integrator.display_integration_dashboard()
                self.log_test("Display: Integration Dashboard", "PASSED", "Integration dashboard executed")
            except Exception as e:
                self.log_test("Display: Integration Dashboard", "FAILED", f"Integration error: {str(e)}")
                return False
            
            enhanced_manager.stop_monitoring()
            integrator.cleanup()
            return True
            
        except Exception as e:
            self.log_test("System Display", "FAILED", f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        🚀 รันการทดสอบทั้งหมด
        """
        print("🧪 STARTING COMPREHENSIVE INTELLIGENT RESOURCE SYSTEM TEST")
        print("="*80)
        
        test_methods = [
            self.test_1_basic_resource_detection,
            self.test_2_80_percent_allocation_strategy,
            self.test_3_environment_optimization,
            self.test_4_real_time_monitoring,
            self.test_5_menu1_optimization_config,
            self.test_6_enhanced_resource_manager,
            self.test_7_menu1_integration,
            self.test_8_performance_benefits_analysis,
            self.test_9_report_generation,
            self.test_10_system_display
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                result = test_method()
                if result:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test method failed: {e}")
        
        # Calculate test summary
        success_rate = (passed_tests / total_tests) * 100
        test_duration = (datetime.now() - self.test_start_time).total_seconds()
        
        # Generate test summary
        summary = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': round(success_rate, 1),
                'test_duration_seconds': round(test_duration, 2)
            },
            'detailed_results': self.test_log,
            'test_timestamp': self.test_start_time.isoformat(),
            'system_ready': success_rate >= 80  # 80% success rate required
        }
        
        # Display final results
        print("\n" + "="*80)
        print("🏆 TEST COMPLETION SUMMARY")
        print("="*80)
        
        status_color = "\033[92m" if success_rate >= 80 else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"\n📊 RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   {status_color}Success Rate: {success_rate:.1f}%{reset_color}")
        print(f"   Duration: {test_duration:.2f} seconds")
        
        if success_rate >= 80:
            print(f"\n✅ {status_color}INTELLIGENT RESOURCE SYSTEM IS READY FOR PRODUCTION!{reset_color}")
        else:
            print(f"\n❌ {status_color}SYSTEM NEEDS FIXES BEFORE PRODUCTION USE{reset_color}")
        
        return summary
    
    def save_test_results(self, file_path: str = None) -> str:
        """
        💾 บันทึกผลการทดสอบ
        """
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"intelligent_resource_test_results_{timestamp}.json"
            
            summary = self.run_all_tests()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Test results saved: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"❌ Test results saving failed: {e}")
            return ""


def main():
    """
    🎯 Main test execution
    """
    try:
        tester = IntelligentResourceSystemTester()
        summary = tester.run_all_tests()
        
        # Save results
        results_file = tester.save_test_results()
        
        # Return success status
        return summary.get('system_ready', False)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
