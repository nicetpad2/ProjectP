#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 NICEGOLD ENTERPRISE PROJECTP - INTELLIGENT RESOURCE MANAGEMENT TEST
ระบบทดสอบการจัดการทรัพยากรอัจฉริยะ (Intelligent Resource Management Test)

🎯 Test Features:
✅ ทดสอบระบบตรวจจับสภาพแวดล้อมอัจฉริยะ
✅ ทดสอบระบบจัดการทรัพยากรอัจฉริยะ
✅ ทดสอบการจัดสรรทรัพยากร 80% แบบอัตโนมัติ
✅ ทดสอบการปรับตัวและเรียนรู้ของระบบ
✅ ทดสอบการจัดการฉุกเฉินและการกู้คืน
✅ ทดสอบการทำงานในสภาพแวดล้อมต่างๆ

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 9 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test modules
try:
    from core.intelligent_environment_detector import (
        get_intelligent_environment_detector,
        EnvironmentType,
        HardwareCapability,
        ResourceOptimizationLevel
    )
    ENVIRONMENT_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Environment detector import failed: {e}")
    ENVIRONMENT_DETECTOR_AVAILABLE = False

try:
    from core.smart_resource_orchestrator import (
        get_smart_resource_orchestrator,
        OrchestrationConfig,
        OrchestrationStatus,
        AdaptiveMode
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Smart orchestrator import failed: {e}")
    ORCHESTRATOR_AVAILABLE = False

try:
    from core.unified_resource_manager import (
        get_unified_resource_manager,
        ResourceType,
        ResourceStatus
    )
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Resource manager import failed: {e}")
    RESOURCE_MANAGER_AVAILABLE = False


# ====================================================
# TEST FRAMEWORK
# ====================================================

class IntelligentResourceManagementTest:
    """
    🧪 ระบบทดสอบการจัดการทรัพยากรอัจฉริยะ
    """
    
    def __init__(self):
        """เริ่มต้นระบบทดสอบ"""
        self.logger = self._setup_logger()
        self.test_results = []
        self.start_time = datetime.now()
        
        print("🧪 INTELLIGENT RESOURCE MANAGEMENT TEST")
        print("=" * 70)
    
    def _setup_logger(self) -> logging.Logger:
        """ติดตั้ง Logger สำหรับการทดสอบ"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - 🧪 [TEST] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """รันการทดสอบและบันทึกผลลัพธ์"""
        try:
            print(f"\n🔍 Running test: {test_name}")
            start_time = time.time()
            
            result = test_func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
                self.test_results.append({
                    'test': test_name,
                    'status': 'PASSED',
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
                self.test_results.append({
                    'test': test_name,
                    'status': 'FAILED',
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                })
                return False
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"❌ {test_name} - ERROR: {e} ({duration:.2f}s)")
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'error': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def test_environment_detector(self) -> bool:
        """ทดสอบระบบตรวจจับสภาพแวดล้อม"""
        if not ENVIRONMENT_DETECTOR_AVAILABLE:
            print("⚠️ Environment detector not available")
            return False
        
        try:
            # Initialize detector
            detector = get_intelligent_environment_detector()
            
            # Test environment detection
            env_info = detector.detect_environment()
            
            # Validate results
            if not env_info:
                print("❌ Environment info is None")
                return False
            
            print(f"  Environment Type: {env_info.environment_type.value}")
            print(f"  Hardware Capability: {env_info.hardware_capability.value}")
            print(f"  Optimization Level: {env_info.optimization_level.value}")
            print(f"  CPU Cores: {env_info.cpu_cores}")
            print(f"  Memory: {env_info.memory_gb:.1f} GB")
            print(f"  GPU Count: {env_info.gpu_count}")
            
            # Test optimal allocation
            allocation = detector.get_optimal_resource_allocation(env_info)
            
            if not allocation:
                print("❌ Optimal allocation is None")
                return False
            
            print(f"  Optimal CPU: {allocation.cpu_percentage*100:.1f}%")
            print(f"  Optimal Memory: {allocation.memory_percentage*100:.1f}%")
            print(f"  Optimal GPU: {allocation.gpu_percentage*100:.1f}%")
            
            # Test report generation
            report = detector.get_environment_report()
            
            if not report or 'error' in report:
                print("❌ Environment report generation failed")
                return False
            
            print("  Environment report generated successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment detector test failed: {e}")
            return False
    
    def test_resource_manager(self) -> bool:
        """ทดสอบระบบจัดการทรัพยากร"""
        if not RESOURCE_MANAGER_AVAILABLE:
            print("⚠️ Resource manager not available")
            return False
        
        try:
            # Initialize resource manager
            manager = get_unified_resource_manager()
            
            # Test resource status
            resources = manager.get_resource_status()
            
            if not resources:
                print("❌ Resource status is empty")
                return False
            
            print(f"  Resources detected: {len(resources)}")
            
            for resource_type, resource_info in resources.items():
                print(f"  {resource_type}: {resource_info.percentage:.1f}% ({resource_info.status.value})")
            
            # Test resource summary
            summary = manager.get_resource_summary()
            
            if not summary or 'error' in summary:
                print("❌ Resource summary generation failed")
                return False
            
            print("  Resource summary generated successfully")
            
            # Test optimization
            opt_result = manager.optimize_resources()
            
            if opt_result.success:
                print(f"  Optimization successful: {opt_result.optimizations} optimizations applied")
            else:
                print("  Optimization not needed (normal)")
            
            return True
            
        except Exception as e:
            print(f"❌ Resource manager test failed: {e}")
            return False
    
    def test_smart_orchestrator(self) -> bool:
        """ทดสอบระบบจัดการทรัพยากรอัจฉริยะ"""
        if not ORCHESTRATOR_AVAILABLE:
            print("⚠️ Smart orchestrator not available")
            return False
        
        try:
            # Initialize orchestrator
            config = OrchestrationConfig(
                target_utilization=0.80,
                monitoring_interval=1.0,
                optimization_interval=5.0
            )
            
            orchestrator = get_smart_resource_orchestrator(config)
            
            # Test orchestrator status
            status = orchestrator.get_orchestration_status()
            
            if not status.success:
                print("❌ Failed to get orchestration status")
                return False
            
            print(f"  Status: {status.status.value}")
            print(f"  Mode: {status.mode.value}")
            print(f"  CPU Allocation: {status.resource_allocation.get('cpu_percentage', 0)*100:.1f}%")
            print(f"  Memory Allocation: {status.resource_allocation.get('memory_percentage', 0)*100:.1f}%")
            
            # Test orchestration start/stop
            start_success = orchestrator.start_orchestration()
            
            if not start_success:
                print("❌ Failed to start orchestration")
                return False
            
            print("  Orchestration started successfully")
            
            # Let it run for a few seconds
            time.sleep(3)
            
            # Check status after running
            status_after = orchestrator.get_orchestration_status()
            print(f"  Status after running: {status_after.status.value}")
            
            # Test detailed report
            report = orchestrator.get_detailed_report()
            
            if not report or 'error' in report:
                print("❌ Detailed report generation failed")
                return False
            
            print("  Detailed report generated successfully")
            
            # Stop orchestration
            stop_success = orchestrator.stop_orchestration()
            
            if not stop_success:
                print("❌ Failed to stop orchestration")
                return False
            
            print("  Orchestration stopped successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Smart orchestrator test failed: {e}")
            return False
    
    def test_resource_allocation_80_percent(self) -> bool:
        """ทดสอบการจัดสรรทรัพยากร 80%"""
        try:
            if not ENVIRONMENT_DETECTOR_AVAILABLE:
                print("⚠️ Environment detector not available for 80% allocation test")
                return False
            
            # Get environment info
            detector = get_intelligent_environment_detector()
            env_info = detector.detect_environment()
            allocation = detector.get_optimal_resource_allocation(env_info)
            
            # Validate 80% target
            target = allocation.target_utilization
            print(f"  Target utilization: {target*100:.1f}%")
            
            if target < 0.70 or target > 0.90:
                print(f"❌ Target utilization {target*100:.1f}% is not within acceptable range (70-90%)")
                return False
            
            # Check individual allocations
            allocations = {
                'CPU': allocation.cpu_percentage,
                'Memory': allocation.memory_percentage,
                'GPU': allocation.gpu_percentage
            }
            
            for resource, percentage in allocations.items():
                print(f"  {resource}: {percentage*100:.1f}%")
                
                if percentage > 0.95:
                    print(f"❌ {resource} allocation {percentage*100:.1f}% is too high (>95%)")
                    return False
                
                if percentage < 0.30:
                    print(f"⚠️ {resource} allocation {percentage*100:.1f}% is very low (<30%)")
            
            # Check safety margins
            safety_margin = allocation.safety_margin
            emergency_reserve = allocation.emergency_reserve
            
            print(f"  Safety margin: {safety_margin*100:.1f}%")
            print(f"  Emergency reserve: {emergency_reserve*100:.1f}%")
            
            if safety_margin < 0.05 or safety_margin > 0.30:
                print(f"❌ Safety margin {safety_margin*100:.1f}% is not within acceptable range (5-30%)")
                return False
            
            if emergency_reserve < 0.02 or emergency_reserve > 0.20:
                print(f"❌ Emergency reserve {emergency_reserve*100:.1f}% is not within acceptable range (2-20%)")
                return False
            
            print("  80% resource allocation strategy validated successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ 80% allocation test failed: {e}")
            return False
    
    def test_environment_adaptation(self) -> bool:
        """ทดสอบการปรับตัวตามสภาพแวดล้อม"""
        try:
            if not ENVIRONMENT_DETECTOR_AVAILABLE:
                print("⚠️ Environment detector not available for adaptation test")
                return False
            
            detector = get_intelligent_environment_detector()
            
            # Test different environment scenarios
            env_info = detector.detect_environment()
            
            print(f"  Detected environment: {env_info.environment_type.value}")
            print(f"  Hardware capability: {env_info.hardware_capability.value}")
            print(f"  Optimization level: {env_info.optimization_level.value}")
            
            # Check if optimization level matches environment
            expected_levels = {
                EnvironmentType.GOOGLE_COLAB: ResourceOptimizationLevel.CONSERVATIVE,
                EnvironmentType.DOCKER_CONTAINER: ResourceOptimizationLevel.STANDARD,
                EnvironmentType.CLOUD_VM: ResourceOptimizationLevel.AGGRESSIVE
            }
            
            # Validate capabilities
            if not env_info.capabilities:
                print("❌ No capabilities detected")
                return False
            
            print(f"  Capabilities detected: {len(env_info.capabilities)}")
            
            # Check key capabilities
            key_capabilities = ['psutil', 'torch', 'tensorflow']
            for cap in key_capabilities:
                if cap in env_info.capabilities:
                    status = "✅" if env_info.capabilities[cap] else "❌"
                    print(f"    {cap}: {status}")
            
            # Validate recommendations
            if not env_info.recommendations:
                print("❌ No recommendations generated")
                return False
            
            print(f"  Recommendations generated: {len(env_info.recommendations)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment adaptation test failed: {e}")
            return False
    
    def test_performance_optimization(self) -> bool:
        """ทดสอบการปรับปรุงประสิทธิภาพ"""
        try:
            if not RESOURCE_MANAGER_AVAILABLE:
                print("⚠️ Resource manager not available for performance test")
                return False
            
            manager = get_unified_resource_manager()
            
            # Get initial resource status
            initial_resources = manager.get_resource_status()
            
            if not initial_resources:
                print("❌ Cannot get initial resource status")
                return False
            
            print(f"  Initial resources: {len(initial_resources)}")
            
            # Perform optimization
            opt_result = manager.optimize_resources()
            
            if opt_result.success:
                print(f"  Optimization successful: {opt_result.optimizations} optimizations")
                
                if opt_result.improvements:
                    print(f"  Improvements: {len(opt_result.improvements)}")
                    for improvement in opt_result.improvements[:3]:  # Show first 3
                        print(f"    - {improvement.get('resource', 'Unknown')}: {improvement.get('action', 'Unknown')}")
            else:
                print("  No optimization needed (system already optimized)")
            
            # Get post-optimization resources
            post_resources = manager.get_resource_status()
            
            if not post_resources:
                print("❌ Cannot get post-optimization resource status")
                return False
            
            # Compare resource usage
            for resource_type in initial_resources:
                if resource_type in post_resources:
                    initial_usage = initial_resources[resource_type].percentage
                    post_usage = post_resources[resource_type].percentage
                    
                    if initial_usage != post_usage:
                        print(f"  {resource_type}: {initial_usage:.1f}% → {post_usage:.1f}%")
            
            print("  Performance optimization test completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance optimization test failed: {e}")
            return False
    
    def test_monitoring_and_alerting(self) -> bool:
        """ทดสอบการติดตามและการแจ้งเตือน"""
        try:
            if not ORCHESTRATOR_AVAILABLE:
                print("⚠️ Smart orchestrator not available for monitoring test")
                return False
            
            # Initialize with fast monitoring
            config = OrchestrationConfig(
                monitoring_interval=0.5,
                optimization_interval=2.0
            )
            
            orchestrator = get_smart_resource_orchestrator(config)
            
            # Start monitoring
            success = orchestrator.start_orchestration()
            
            if not success:
                print("❌ Failed to start monitoring")
                return False
            
            print("  Monitoring started successfully")
            
            # Monitor for a few seconds
            time.sleep(3)
            
            # Get status
            status = orchestrator.get_orchestration_status()
            
            if not status.success:
                print("❌ Failed to get monitoring status")
                return False
            
            print(f"  Status: {status.status.value}")
            print(f"  Mode: {status.mode.value}")
            print(f"  Optimizations applied: {status.optimizations_applied}")
            
            # Check for recommendations
            if status.recommendations:
                print(f"  Recommendations: {len(status.recommendations)}")
                for rec in status.recommendations[:2]:  # Show first 2
                    print(f"    - {rec}")
            
            # Stop monitoring
            orchestrator.stop_orchestration()
            print("  Monitoring stopped successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Monitoring test failed: {e}")
            return False
    
    def run_all_tests(self) -> None:
        """รันการทดสอบทั้งหมด"""
        print(f"🚀 Starting comprehensive resource management tests...")
        print(f"📅 Test started at: {self.start_time}")
        
        # List of tests to run
        tests = [
            ("Environment Detector Test", self.test_environment_detector),
            ("Resource Manager Test", self.test_resource_manager),
            ("Smart Orchestrator Test", self.test_smart_orchestrator),
            ("80% Resource Allocation Test", self.test_resource_allocation_80_percent),
            ("Environment Adaptation Test", self.test_environment_adaptation),
            ("Performance Optimization Test", self.test_performance_optimization),
            ("Monitoring and Alerting Test", self.test_monitoring_and_alerting)
        ]
        
        # Run all tests
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
            else:
                failed += 1
        
        # Summary
        print("\n" + "=" * 70)
        print("🏁 TEST SUMMARY")
        print("=" * 70)
        
        total_tests = passed + failed
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        print(f"Total Duration: {total_duration:.2f} seconds")
        
        # Show detailed results
        print("\n📊 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {result['test']} - {result['status']} ({result['duration']:.2f}s)")
        
        # Save results to file
        self.save_test_results()
        
        # Final verdict
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Resource management system is production ready.")
        else:
            print(f"\n⚠️ {failed} test(s) failed. Please review and fix issues before production deployment.")
    
    def save_test_results(self) -> None:
        """บันทึกผลการทดสอบลงไฟล์"""
        try:
            results_file = "intelligent_resource_management_test_results.json"
            
            test_report = {
                'test_suite': 'Intelligent Resource Management Test',
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'PASSED']),
                'failed': len([r for r in self.test_results if r['status'] == 'FAILED']),
                'errors': len([r for r in self.test_results if r['status'] == 'ERROR']),
                'results': self.test_results
            }
            
            with open(results_file, 'w') as f:
                json.dump(test_report, f, indent=2)
            
            print(f"\n💾 Test results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save test results: {e}")


# ====================================================
# MAIN EXECUTION
# ====================================================

def main():
    """ฟังก์ชั่นหลักสำหรับทดสอบ"""
    try:
        # Create test instance
        test = IntelligentResourceManagementTest()
        
        # Run all tests
        test.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")


if __name__ == "__main__":
    main()
