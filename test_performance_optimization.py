#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ENTERPRISE PERFORMANCE TEST
Direct test of the optimized system to resolve CPU 100% and Memory 30%+ issues

🎯 MISSION: Test the performance optimization system directly
⚡ GOAL: Demonstrate reduced resource usage with maintained accuracy
🛡️ METHOD: Run optimized pipeline on real data with resource monitoring
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Environment optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

# Import optimization engine
try:
    from nicegold_resource_optimization_engine import NiceGoldResourceOptimizationEngine
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Optimization engine not available: {e}")
    OPTIMIZATION_AVAILABLE = False
    sys.exit(1)

# Import data processing
try:
    from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Data processor not available: {e}")
    DATA_PROCESSOR_AVAILABLE = False


class PerformanceTestManager:
    """🧪 Performance Test Manager"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.resource_logs = []
        self.test_results = {}
        
        # Initialize optimization engine
        if OPTIMIZATION_AVAILABLE:
            self.optimization_engine = NiceGoldResourceOptimizationEngine()
            print("🚀 NICEGOLD Resource Optimization Engine initialized")
        else:
            self.optimization_engine = None
            
        # Initialize data processor if available
        if DATA_PROCESSOR_AVAILABLE:
            self.data_processor = ElliottWaveDataProcessor()
            print("📊 Data processor initialized")
        else:
            self.data_processor = None
    
    def monitor_resources(self, stage: str) -> dict:
        """Monitor system resources"""
        usage = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        self.resource_logs.append(usage)
        return usage
    
    def print_resource_status(self, stage: str):
        """Print current resource status"""
        usage = self.monitor_resources(stage)
        print(f"📊 {stage}: CPU {usage['cpu_percent']:.1f}%, Memory {usage['memory_percent']:.1f}% ({usage['memory_used_gb']:.1f}GB)")
    
    def load_test_data(self) -> tuple:
        """Load real data for testing"""
        print("\n🔄 Loading real market data for testing...")
        self.print_resource_status("Data Loading Start")
        
        try:
            # Load real data using data processor
            if self.data_processor:
                data = self.data_processor.load_real_data()
                if data is not None and len(data) > 0:
                    print(f"✅ Loaded {len(data):,} rows of real data")
                    
                    # Create features
                    print("⚙️ Creating Elliott Wave features...")
                    features = self.data_processor.create_elliott_wave_features(data)
                    print(f"✅ Created {features.shape[1]} features")
                    
                    # Prepare ML data
                    print("🎯 Preparing ML data...")
                    X, y = self.data_processor.prepare_ml_data(features)
                    print(f"✅ Prepared ML data: {X.shape[0]} samples, {X.shape[1]} features")
                    
                    self.print_resource_status("Data Loading Complete")
                    return X, y
            
            # Fallback: create synthetic test data
            print("⚠️ Data processor not available, creating synthetic test data...")
            n_samples = 50000  # Large enough to test performance
            n_features = 50
            
            np.random.seed(42)
            X = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            y = pd.Series(np.random.binomial(1, 0.3, n_samples))
            
            print(f"✅ Created synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
            self.print_resource_status("Data Loading Complete")
            return X, y
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            raise
    
    def test_optimization_engine(self, X: pd.DataFrame, y: pd.Series):
        """Test the optimization engine"""
        print("\n🚀 Testing NICEGOLD Resource Optimization Engine...")
        self.print_resource_status("Optimization Test Start")
        
        start_time = time.time()
        
        try:
            # Run optimized pipeline
            results = self.optimization_engine.execute_optimized_pipeline(X, y)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.print_resource_status("Optimization Test Complete")
            
            # Analyze results
            print(f"\n✅ Optimization test completed in {execution_time:.1f} seconds")
            
            # Print key results
            if 'selected_features' in results:
                print(f"🎯 Selected features: {len(results['selected_features'])}")
            
            if 'feature_selection' in results:
                fs_results = results['feature_selection']
                if 'validation_results' in fs_results:
                    validation = fs_results['validation_results']
                    if 'cv_auc_mean' in validation:
                        print(f"📈 Feature selection AUC: {validation['cv_auc_mean']:.3f}")
            
            if 'ml_protection' in results:
                ml_results = results['ml_protection']
                if 'enterprise_ready' in ml_results:
                    print(f"🛡️ Enterprise ready: {ml_results['enterprise_ready']}")
            
            if 'execution_metrics' in results:
                exec_metrics = results['execution_metrics']
                print(f"⚡ Resource efficient: {exec_metrics.get('resource_efficient', False)}")
                print(f"💾 CPU controlled: {exec_metrics.get('cpu_controlled', False)}")
                print(f"🧠 Memory controlled: {exec_metrics.get('memory_controlled', False)}")
            
            self.test_results['optimization_test'] = {
                'success': True,
                'execution_time': execution_time,
                'results': results
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Optimization test failed: {e}")
            self.test_results['optimization_test'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_individual_components(self, X: pd.DataFrame, y: pd.Series):
        """Test individual optimized components"""
        print("\n🔧 Testing individual optimized components...")
        
        # Test feature selector
        print("\n⚡ Testing optimized feature selector...")
        self.print_resource_status("Feature Selection Start")
        
        try:
            start_time = time.time()
            selected_features, fs_results = self.optimization_engine.feature_selector.enterprise_feature_selection(X, y)
            end_time = time.time()
            
            print(f"✅ Feature selection completed in {end_time - start_time:.1f}s")
            print(f"🎯 Selected {len(selected_features)} features")
            
            if 'validation_results' in fs_results:
                validation = fs_results['validation_results']
                if 'cv_auc_mean' in validation:
                    print(f"📈 AUC: {validation['cv_auc_mean']:.3f}")
            
            self.print_resource_status("Feature Selection Complete")
            
            # Test ML protection
            print("\n🛡️ Testing optimized ML protection...")
            self.print_resource_status("ML Protection Start")
            
            start_time = time.time()
            protection_results = self.optimization_engine.ml_protection.enterprise_protection_analysis(X[selected_features], y)
            end_time = time.time()
            
            print(f"✅ ML protection completed in {end_time - start_time:.1f}s")
            print(f"🛡️ Enterprise ready: {protection_results.get('enterprise_ready', False)}")
            
            self.print_resource_status("ML Protection Complete")
            
            self.test_results['component_tests'] = {
                'feature_selection': {'success': True, 'features': len(selected_features)},
                'ml_protection': {'success': True, 'enterprise_ready': protection_results.get('enterprise_ready', False)}
            }
            
        except Exception as e:
            print(f"❌ Component test failed: {e}")
            self.test_results['component_tests'] = {'success': False, 'error': str(e)}
    
    def analyze_performance_improvement(self):
        """Analyze performance improvement"""
        print("\n📊 Performance Analysis...")
        
        if not self.resource_logs:
            print("⚠️ No resource logs available for analysis")
            return
        
        # Find max resource usage
        max_cpu = max(log['cpu_percent'] for log in self.resource_logs)
        max_memory = max(log['memory_percent'] for log in self.resource_logs)
        avg_cpu = sum(log['cpu_percent'] for log in self.resource_logs) / len(self.resource_logs)
        avg_memory = sum(log['memory_percent'] for log in self.resource_logs) / len(self.resource_logs)
        
        print(f"🔥 Peak CPU usage: {max_cpu:.1f}%")
        print(f"🧠 Peak Memory usage: {max_memory:.1f}%")
        print(f"⚡ Average CPU usage: {avg_cpu:.1f}%")
        print(f"💾 Average Memory usage: {avg_memory:.1f}%")
        
        # Check if we achieved our goals
        cpu_goal_achieved = max_cpu < 80  # Target: Keep CPU under 80%
        memory_goal_achieved = max_memory < 70  # Target: Keep memory under 70%
        
        print(f"\n🎯 Performance Goals:")
        print(f"   CPU < 80%: {'✅ ACHIEVED' if cpu_goal_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Memory < 70%: {'✅ ACHIEVED' if memory_goal_achieved else '❌ NOT ACHIEVED'}")
        
        return {
            'max_cpu': max_cpu,
            'max_memory': max_memory,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'cpu_goal_achieved': cpu_goal_achieved,
            'memory_goal_achieved': memory_goal_achieved
        }
    
    def generate_report(self):
        """Generate performance test report"""
        print("\n" + "="*60)
        print("🎯 NICEGOLD PERFORMANCE OPTIMIZATION TEST REPORT")
        print("="*60)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"⏱️ Total test duration: {total_time:.1f} seconds")
        
        # Performance analysis
        perf_analysis = self.analyze_performance_improvement()
        
        # Test results summary
        print(f"\n📋 Test Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        # Overall assessment
        all_tests_passed = all(result.get('success', False) for result in self.test_results.values())
        performance_goals_met = (
            perf_analysis['cpu_goal_achieved'] and 
            perf_analysis['memory_goal_achieved']
        ) if perf_analysis else False
        
        overall_success = all_tests_passed and performance_goals_met
        
        print(f"\n🏆 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
        
        if overall_success:
            print("🎉 Performance optimization successfully resolved CPU and memory issues!")
        else:
            print("⚠️ Performance optimization needs further tuning.")
        
        print("="*60)
        
        return {
            'overall_success': overall_success,
            'performance_analysis': perf_analysis,
            'test_results': self.test_results,
            'total_duration': total_time
        }


def main():
    """Main test execution"""
    print("🚀 NICEGOLD ENTERPRISE PERFORMANCE TEST")
    print("="*50)
    print("🎯 Testing resource optimization to resolve CPU 100% and Memory 30%+ issues")
    print("")
    
    # Initialize test manager
    test_manager = PerformanceTestManager()
    
    try:
        # Load test data
        X, y = test_manager.load_test_data()
        
        # Test optimization engine
        test_manager.test_optimization_engine(X, y)
        
        # Test individual components
        test_manager.test_individual_components(X, y)
        
        # Generate final report
        report = test_manager.generate_report()
        
        # Return success status
        return 0 if report['overall_success'] else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
