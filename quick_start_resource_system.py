#!/usr/bin/env python3
"""
⚡ QUICK START INTELLIGENT RESOURCE SYSTEM
=========================================

Quick validation และเริ่มต้นใช้งานระบบจัดการทรัพยากรอัจฉริยะ
สำหรับ NICEGOLD ProjectP Elliott Wave Pipeline

🚀 Quick Features:
- One-click system validation
- Instant resource check
- Direct Menu 1 with optimization
- Quick performance test
- Fast system readiness verification
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)

# Import resource management components
try:
    from core.intelligent_resource_manager import initialize_intelligent_resources
    from core.menu1_resource_integration import create_menu1_resource_integrator
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class QuickStartResourceSystem:
    """
    ⚡ Quick Start for Intelligent Resource System
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.resource_manager = None
        self.integrator = None
        
    def quick_system_check(self) -> bool:
        """
        🔍 Quick system resource check
        """
        try:
            print("🔍 Quick System Check...")
            
            # Initialize basic resource manager
            self.resource_manager = initialize_intelligent_resources()
            
            # Quick validation
            system_info = self.resource_manager.system_info
            
            # Check CPU
            cpu_cores = system_info.get('cpu', {}).get('logical_cores', 0)
            if cpu_cores < 2:
                print(f"⚠️ Warning: Only {cpu_cores} CPU cores detected")
                return False
            
            # Check Memory
            memory_gb = system_info.get('memory', {}).get('total_gb', 0)
            if memory_gb < 4:
                print(f"⚠️ Warning: Only {memory_gb:.1f}GB RAM detected")
                return False
            
            # Display quick summary
            print(f"✅ CPU: {cpu_cores} cores")
            print(f"✅ RAM: {memory_gb:.1f} GB")
            
            gpu_available = system_info.get('gpu', {}).get('cuda_available', False)
            print(f"{'✅' if gpu_available else '💻'} GPU: {'CUDA Available' if gpu_available else 'CPU-only mode'}")
            
            return True
            
        except Exception as e:
            print(f"❌ System check failed: {e}")
            return False
    
    def quick_allocation_test(self) -> bool:
        """
        ⚡ Quick allocation strategy test
        """
        try:
            print("\n⚡ Testing 80% Allocation Strategy...")
            
            if not self.resource_manager:
                return False
            
            # Check resource allocation
            resource_config = self.resource_manager.resource_config
            
            cpu_config = resource_config.get('cpu', {})
            memory_config = resource_config.get('memory', {})
            optimization = resource_config.get('optimization', {})
            
            allocated_threads = cpu_config.get('allocated_threads', 0)
            allocated_memory = memory_config.get('allocated_gb', 0)
            batch_size = optimization.get('batch_size', 0)
            
            if allocated_threads > 0 and allocated_memory > 0 and batch_size > 0:
                print(f"✅ Threads: {allocated_threads}")
                print(f"✅ Memory: {allocated_memory:.1f} GB")
                print(f"✅ Batch Size: {batch_size}")
                return True
            else:
                print(f"❌ Allocation calculation failed")
                return False
                
        except Exception as e:
            print(f"❌ Allocation test failed: {e}")
            return False
    
    def quick_monitoring_test(self) -> bool:
        """
        📊 Quick monitoring test
        """
        try:
            print("\n📊 Testing Real-time Monitoring...")
            
            if not self.resource_manager:
                return False
            
            # Start monitoring
            self.resource_manager.start_monitoring(interval=0.5)
            
            # Wait briefly
            time.sleep(2)
            
            # Check monitoring
            if self.resource_manager.monitoring_active:
                data_points = len(self.resource_manager.performance_data)
                print(f"✅ Monitoring active: {data_points} data points")
                
                # Stop monitoring
                self.resource_manager.stop_monitoring()
                return True
            else:
                print(f"❌ Monitoring not active")
                return False
                
        except Exception as e:
            print(f"❌ Monitoring test failed: {e}")
            return False
    
    def quick_menu1_integration_test(self) -> bool:
        """
        🌊 Quick Menu 1 integration test
        """
        try:
            print("\n🌊 Testing Menu 1 Integration...")
            
            # Create mock Menu 1
            class QuickMockMenu1:
                def __init__(self):
                    self.config = {}
            
            mock_menu1 = QuickMockMenu1()
            
            # Create integrator
            self.integrator = create_menu1_resource_integrator()
            
            # Test integration
            success = self.integrator.integrate_with_menu1(mock_menu1)
            
            if success:
                print(f"✅ Menu 1 integration successful")
                
                # Quick stage test
                self.integrator.start_stage('data_loading')
                time.sleep(0.5)
                summary = self.integrator.end_stage('data_loading', {'duration': 5.0})
                
                if summary:
                    print(f"✅ Stage monitoring working")
                    return True
                else:
                    print(f"❌ Stage monitoring failed")
                    return False
            else:
                print(f"❌ Menu 1 integration failed")
                return False
                
        except Exception as e:
            print(f"❌ Menu 1 integration test failed: {e}")
            return False
    
    def quick_performance_validation(self) -> bool:
        """
        🚀 Quick performance validation
        """
        try:
            print("\n🚀 Quick Performance Validation...")
            
            if not self.resource_manager:
                return False
            
            # Apply optimization
            optimization_success = self.resource_manager.apply_resource_optimization()
            
            if optimization_success:
                print(f"✅ Resource optimization applied")
            else:
                print(f"⚠️ Resource optimization partially applied")
            
            # Get Menu 1 config
            menu1_config = self.resource_manager.get_menu1_optimization_config()
            
            if menu1_config and len(menu1_config) >= 5:
                print(f"✅ Menu 1 optimization config ready")
                return True
            else:
                print(f"❌ Menu 1 config incomplete")
                return False
                
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return False
    
    def display_quick_summary(self, test_results: list) -> None:
        """
        📋 Display quick summary
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\n" + "="*60)
        print(f"⚡ QUICK START SUMMARY")
        print(f"="*60)
        
        print(f"📊 Results:")
        print(f"   • Tests Passed: {passed_tests}/{total_tests}")
        print(f"   • Success Rate: {success_rate:.1f}%")
        print(f"   • Duration: {duration:.1f} seconds")
        
        if success_rate >= 80:
            print(f"\n✅ SYSTEM READY FOR PRODUCTION!")
            print(f"   Your Intelligent Resource Management System is ready")
            print(f"   to enhance NICEGOLD ProjectP Elliott Wave Pipeline.")
            
            if self.resource_manager:
                config = self.resource_manager.resource_config
                cpu_threads = config.get('cpu', {}).get('allocated_threads', 0)
                memory_gb = config.get('memory', {}).get('allocated_gb', 0)
                batch_size = config.get('optimization', {}).get('batch_size', 0)
                
                print(f"\n🎯 Optimized Configuration:")
                print(f"   • CPU Threads: {cpu_threads}")
                print(f"   • Memory Allocation: {memory_gb:.1f} GB") 
                print(f"   • Batch Size: {batch_size}")
        else:
            print(f"\n⚠️ SYSTEM NEEDS ATTENTION")
            print(f"   Some tests failed. Please check system requirements.")
        
        print(f"\n🚀 Quick Commands:")
        print(f"   • Full Demo: python demo_intelligent_resource_management.py")
        print(f"   • Full Test: python test_menu1_with_intelligent_resources.py")
        print(f"   • Run ProjectP: python ProjectP.py")
    
    def run_quick_start(self) -> bool:
        """
        🚀 รัน Quick Start ทั้งหมด
        """
        print("⚡ INTELLIGENT RESOURCE SYSTEM - QUICK START")
        print("="*60)
        print("Fast validation and setup for resource optimization")
        
        # Run quick tests
        test_methods = [
            ("System Check", self.quick_system_check),
            ("Allocation Test", self.quick_allocation_test),
            ("Monitoring Test", self.quick_monitoring_test),
            ("Menu 1 Integration", self.quick_menu1_integration_test),
            ("Performance Validation", self.quick_performance_validation)
        ]
        
        test_results = []
        
        for test_name, test_func in test_methods:
            try:
                result = test_func()
                test_results.append(result)
                
                if not result:
                    print(f"❌ {test_name} failed")
                
            except Exception as e:
                print(f"❌ {test_name} error: {e}")
                test_results.append(False)
        
        # Display summary
        self.display_quick_summary(test_results)
        
        # Cleanup
        if self.resource_manager:
            self.resource_manager.stop_monitoring()
        if self.integrator:
            self.integrator.cleanup()
        
        # Return overall success
        success_rate = (sum(test_results) / len(test_results)) * 100
        return success_rate >= 80
    
    def get_quick_menu1_config(self) -> dict:
        """
        🌊 รับ configuration สำหรับ Menu 1 แบบรวดเร็ว
        """
        if self.resource_manager:
            return self.resource_manager.get_menu1_optimization_config()
        return {}


def quick_start_intelligent_resources() -> bool:
    """
    ⚡ Function สำหรับเริ่มต้นใช้งานแบบรวดเร็ว
    """
    try:
        quick_starter = QuickStartResourceSystem()
        return quick_starter.run_quick_start()
    except Exception as e:
        print(f"❌ Quick start failed: {e}")
        return False


def get_optimized_menu1_config() -> dict:
    """
    🎯 Function สำหรับรับ configuration ที่ปรับแต่งแล้วสำหรับ Menu 1
    """
    try:
        quick_starter = QuickStartResourceSystem()
        if quick_starter.quick_system_check():
            return quick_starter.get_quick_menu1_config()
        return {}
    except Exception as e:
        print(f"❌ Config retrieval failed: {e}")
        return {}


def main():
    """
    🎯 Main quick start execution
    """
    try:
        print("⚡ Starting Intelligent Resource System Quick Start...")
        
        success = quick_start_intelligent_resources()
        
        if success:
            print(f"\n🎉 Quick Start Completed Successfully!")
            print(f"Your system is ready for enhanced Menu 1 performance!")
        else:
            print(f"\n⚠️ Quick Start completed with issues.")
            print(f"Run full tests for detailed diagnostics.")
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Quick start interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Quick start execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
