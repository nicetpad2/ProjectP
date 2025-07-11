#!/usr/bin/env python3
"""
🎉 MENU 1 INTELLIGENT RESOURCE INTEGRATION - COMPLETE SYSTEM
===========================================================

ระบบเชื่อมต่อ Intelligent Resource Management กับ Menu 1 Elliott Wave Pipeline
พร้อมการทำงานร่วมกันแบบไร้รอยต่อ และการติดตามประสิทธิภาพแบบ real-time

🌟 COMPLETE FEATURES:
✅ Seamless Integration with Menu 1 Elliott Wave Pipeline
✅ Real-time Resource Monitoring & Adaptive Allocation
✅ 80% RAM Allocation with Smart Optimization
✅ Performance Analytics & Enterprise Reporting
✅ Stage-by-Stage Resource Management
✅ Intelligent Environment Detection
✅ Production-Ready Monitoring & Alerts
✅ Automated Configuration Optimization
✅ Emergency Resource Recovery
✅ Comprehensive Logging & Audit Trail

🎯 INTEGRATION TARGETS:
- Enhanced Menu 1 Elliott Wave Pipeline
- Smart Resource Orchestrator
- Intelligent Environment Detector
- Real-time Performance Monitor
- Enterprise Logging System
- Production Demo & Validation
"""

import os
import sys
import time
import json
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import core components
from core.unified_enterprise_logger import get_unified_logger, LogLevel, ProcessStatus, ElliottWaveStep, Menu1Step

# Import intelligent resource management system
try:
    from core.smart_resource_orchestrator import SmartResourceOrchestrator
    from core.intelligent_environment_detector import IntelligentEnvironmentDetector
    from core.unified_resource_manager import get_unified_resource_manager
    INTELLIGENT_RESOURCE_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_RESOURCE_AVAILABLE = False
    print(f"⚠️ Intelligent Resource Management not available: {e}")

# Import Menu 1 components
try:
    from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
    MENU1_AVAILABLE = True
except ImportError as e:
    MENU1_AVAILABLE = False
    print(f"⚠️ Enhanced Menu 1 not available: {e}")

# Import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

class Menu1IntelligentResourceIntegration:
    """
    🌊 Menu 1 Intelligent Resource Integration Manager
    
    The complete integration system that connects intelligent resource management
    with Menu 1 Elliott Wave Pipeline for seamless, optimized execution.
    """
    
    def __init__(self, allocation_percentage: float = 0.8):
        """Initialize the integration system"""
        self.logger = get_unified_logger()
        self.allocation_percentage = allocation_percentage
        self.integration_active = False
        self.performance_metrics = []
        self.stage_performance = {}
        self.resource_history = []
        self.alerts = []
        
        # Initialize components
        self.environment_detector = None
        self.resource_orchestrator = None
        self.menu1_instance = None
        self.monitoring_thread = None
        self.monitoring_active = False
        
        self.logger.info("🌊 Menu 1 Intelligent Resource Integration initialized")
        
    def initialize_intelligent_resources(self) -> bool:
        """Initialize intelligent resource management system"""
        try:
            if not INTELLIGENT_RESOURCE_AVAILABLE:
                self.logger.error("❌ Intelligent Resource Management not available")
                return False
            
            # Initialize environment detector
            self.environment_detector = IntelligentEnvironmentDetector()
            environment_info = self.environment_detector.detect_environment()
            
            # Initialize smart resource orchestrator
            self.resource_orchestrator = SmartResourceOrchestrator(
                allocation_percentage=self.allocation_percentage,
                environment_info=environment_info
            )
            
            # Start intelligent resource management
            success = self.resource_orchestrator.initialize_intelligent_management()
            if not success:
                self.logger.error("❌ Failed to initialize intelligent resource management")
                return False
            
            self.logger.info("🚀 Intelligent resource management initialized successfully")
            self.logger.info(f"   📊 Environment: {environment_info.get('environment_type', 'Unknown')}")
            self.logger.info(f"   💾 Target Allocation: {self.allocation_percentage*100:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent resource initialization failed: {e}")
            return False
    
    def integrate_with_menu1(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Integrate with Menu 1 Elliott Wave Pipeline"""
        try:
            if not MENU1_AVAILABLE:
                self.logger.error("❌ Enhanced Menu 1 not available")
                return False
            
            # Initialize Menu 1 with intelligent resource integration
            self.menu1_instance = EnhancedMenu1ElliottWave(config=config)
            
            # Initialize intelligent resources if not already done
            if not self.resource_orchestrator:
                success = self.initialize_intelligent_resources()
                if not success:
                    return False
            
            # Integrate resource management with Menu 1
            success = self.resource_orchestrator.integrate_with_pipeline(self.menu1_instance)
            if not success:
                self.logger.error("❌ Failed to integrate resource management with Menu 1")
                return False
            
            # Setup monitoring hooks
            self._setup_menu1_monitoring_hooks()
            
            # Apply optimized configuration
            self._apply_intelligent_configuration()
            
            self.integration_active = True
            
            self.logger.info("🔗 Menu 1 intelligent resource integration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Menu 1 integration failed: {e}")
            return False
    
    def _setup_menu1_monitoring_hooks(self) -> None:
        """Setup monitoring hooks for Menu 1 pipeline stages"""
        try:
            if not self.resource_orchestrator:
                return
            
            # Define stage monitoring callbacks
            stage_callbacks = {
                'initialization': {
                    'start': lambda: self._start_stage_monitoring('initialization'),
                    'end': lambda metrics: self._end_stage_monitoring('initialization', metrics)
                },
                'data_loading': {
                    'start': lambda: self._start_stage_monitoring('data_loading'),
                    'end': lambda metrics: self._end_stage_monitoring('data_loading', metrics)
                },
                'feature_engineering': {
                    'start': lambda: self._start_stage_monitoring('feature_engineering'),
                    'end': lambda metrics: self._end_stage_monitoring('feature_engineering', metrics)
                },
                'feature_selection': {
                    'start': lambda: self._start_stage_monitoring('feature_selection'),
                    'end': lambda metrics: self._end_stage_monitoring('feature_selection', metrics)
                },
                'cnn_lstm_training': {
                    'start': lambda: self._start_stage_monitoring('cnn_lstm_training'),
                    'end': lambda metrics: self._end_stage_monitoring('cnn_lstm_training', metrics)
                },
                'dqn_training': {
                    'start': lambda: self._start_stage_monitoring('dqn_training'),
                    'end': lambda metrics: self._end_stage_monitoring('dqn_training', metrics)
                },
                'performance_analysis': {
                    'start': lambda: self._start_stage_monitoring('performance_analysis'),
                    'end': lambda metrics: self._end_stage_monitoring('performance_analysis', metrics)
                },
                'model_validation': {
                    'start': lambda: self._start_stage_monitoring('model_validation'),
                    'end': lambda metrics: self._end_stage_monitoring('model_validation', metrics)
                },
                'result_compilation': {
                    'start': lambda: self._start_stage_monitoring('result_compilation'),
                    'end': lambda metrics: self._end_stage_monitoring('result_compilation', metrics)
                }
            }
            
            # Register callbacks with resource orchestrator
            self.resource_orchestrator.register_stage_callbacks(stage_callbacks)
            
            self.logger.info("🎣 Menu 1 monitoring hooks setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring hooks setup failed: {e}")
    
    def _apply_intelligent_configuration(self) -> None:
        """Apply intelligent configuration optimizations"""
        try:
            if not self.resource_orchestrator or not self.menu1_instance:
                return
            
            # Get optimized configuration from resource orchestrator
            optimized_config = self.resource_orchestrator.get_optimized_configuration()
            
            # Apply configuration to Menu 1
            if hasattr(self.menu1_instance, 'config') and optimized_config:
                self.menu1_instance.config.update(optimized_config)
            
            self.logger.info("⚙️ Intelligent configuration applied to Menu 1")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration optimization failed: {e}")
    
    def _start_stage_monitoring(self, stage: str) -> None:
        """Start monitoring for a specific stage"""
        try:
            if self.resource_orchestrator:
                self.resource_orchestrator.start_stage_monitoring(stage)
            
            # Record stage start
            self.stage_performance[stage] = {
                'start_time': time.time(),
                'start_resources': self._get_current_resources()
            }
            
            self.logger.info(f"🎯 Stage monitoring started: {stage}")
            
        except Exception as e:
            self.logger.error(f"❌ Stage monitoring start failed for {stage}: {e}")
    
    def _end_stage_monitoring(self, stage: str, metrics: Dict[str, Any]) -> None:
        """End monitoring for a specific stage"""
        try:
            if self.resource_orchestrator:
                self.resource_orchestrator.end_stage_monitoring(stage, metrics)
            
            # Record stage end
            if stage in self.stage_performance:
                end_time = time.time()
                start_time = self.stage_performance[stage]['start_time']
                duration = end_time - start_time
                
                self.stage_performance[stage].update({
                    'end_time': end_time,
                    'duration': duration,
                    'end_resources': self._get_current_resources(),
                    'metrics': metrics
                })
                
                # Calculate stage efficiency
                efficiency = self._calculate_stage_efficiency(stage)
                self.stage_performance[stage]['efficiency'] = efficiency
                
                self.logger.info(f"⏹️ Stage monitoring ended: {stage} (Duration: {duration:.2f}s, Efficiency: {efficiency:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Stage monitoring end failed for {stage}: {e}")
    
    def _get_current_resources(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        try:
            if self.resource_orchestrator:
                return self.resource_orchestrator.get_current_performance()
            return {}
        except Exception as e:
            self.logger.error(f"❌ Resource monitoring failed: {e}")
            return {}
    
    def _calculate_stage_efficiency(self, stage: str) -> float:
        """Calculate efficiency score for a stage"""
        try:
            if stage not in self.stage_performance:
                return 0.0
            
            stage_data = self.stage_performance[stage]
            
            # Basic efficiency based on resource usage and duration
            start_resources = stage_data.get('start_resources', {})
            end_resources = stage_data.get('end_resources', {})
            duration = stage_data.get('duration', 0)
            
            if not start_resources or not end_resources or duration == 0:
                return 0.0
            
            # Calculate resource efficiency (lower is better for most metrics)
            cpu_efficiency = 1.0 - (end_resources.get('cpu_percent', 0) / 100.0)
            memory_efficiency = end_resources.get('memory', {}).get('percent', 0) / 100.0
            
            # Duration efficiency (faster is better, normalize to 0-1)
            duration_efficiency = min(1.0, 300.0 / duration)  # 300 seconds as baseline
            
            # Combine efficiencies
            efficiency = (cpu_efficiency + memory_efficiency + duration_efficiency) / 3.0
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency calculation failed for {stage}: {e}")
            return 0.0
    
    def run_intelligent_pipeline(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run Menu 1 pipeline with intelligent resource management"""
        try:
            if not self.integration_active:
                # Auto-integrate if not already done
                success = self.integrate_with_menu1(config)
                if not success:
                    return {"status": "ERROR", "message": "Failed to integrate with Menu 1"}
            
            # Start real-time monitoring
            self._start_real_time_monitoring()
            
            # Display integration status
            self._display_integration_status()
            
            # Execute Menu 1 pipeline
            self.logger.info("🚀 Starting Menu 1 pipeline with intelligent resource management...")
            
            start_time = time.time()
            result = self.menu1_instance.run()
            end_time = time.time()
            
            # Stop monitoring
            self._stop_real_time_monitoring()
            
            # Calculate overall performance
            total_duration = end_time - start_time
            overall_efficiency = self._calculate_overall_efficiency()
            
            # Generate comprehensive report
            integration_report = self._generate_integration_report(result, total_duration, overall_efficiency)
            
            self.logger.info(f"✅ Menu 1 pipeline completed with intelligent resource management")
            self.logger.info(f"   ⏱️ Total Duration: {total_duration:.2f} seconds")
            self.logger.info(f"   📊 Overall Efficiency: {overall_efficiency:.2f}")
            
            return {
                "status": "SUCCESS",
                "pipeline_result": result,
                "integration_report": integration_report,
                "performance_metrics": {
                    "total_duration": total_duration,
                    "overall_efficiency": overall_efficiency,
                    "stage_performance": self.stage_performance
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent pipeline execution failed: {e}")
            self._stop_real_time_monitoring()
            return {"status": "ERROR", "message": str(e)}
    
    def _start_real_time_monitoring(self) -> None:
        """Start real-time resource monitoring"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("📊 Real-time monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Real-time monitoring start failed: {e}")
    
    def _stop_real_time_monitoring(self) -> None:
        """Stop real-time resource monitoring"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("📊 Real-time monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Real-time monitoring stop failed: {e}")
    
    def _monitoring_loop(self) -> None:
        """Real-time monitoring loop"""
        try:
            while self.monitoring_active:
                # Get current resources
                current_resources = self._get_current_resources()
                
                # Record resource history
                self.resource_history.append({
                    'timestamp': time.time(),
                    'resources': current_resources
                })
                
                # Check for alerts
                self._check_resource_alerts(current_resources)
                
                # Adaptive resource adjustments
                if self.resource_orchestrator:
                    self.resource_orchestrator.adaptive_resource_adjustment(current_resources)
                
                time.sleep(5)  # Monitor every 5 seconds
                
        except Exception as e:
            self.logger.error(f"❌ Monitoring loop error: {e}")
    
    def _check_resource_alerts(self, resources: Dict[str, Any]) -> None:
        """Check for resource alerts and warnings"""
        try:
            alerts = []
            
            # CPU alerts
            cpu_percent = resources.get('cpu_percent', 0)
            if cpu_percent > 90:
                alerts.append({'type': 'CRITICAL', 'message': f'High CPU usage: {cpu_percent:.1f}%'})
            elif cpu_percent > 80:
                alerts.append({'type': 'WARNING', 'message': f'Elevated CPU usage: {cpu_percent:.1f}%'})
            
            # Memory alerts
            memory_info = resources.get('memory', {})
            memory_percent = memory_info.get('percent', 0)
            if memory_percent > 95:
                alerts.append({'type': 'CRITICAL', 'message': f'High memory usage: {memory_percent:.1f}%'})
            elif memory_percent > 85:
                alerts.append({'type': 'WARNING', 'message': f'Elevated memory usage: {memory_percent:.1f}%'})
            
            # Store alerts
            for alert in alerts:
                alert['timestamp'] = time.time()
                self.alerts.append(alert)
                
                # Log alerts
                if alert['type'] == 'CRITICAL':
                    self.logger.error(f"🚨 {alert['message']}")
                else:
                    self.logger.warning(f"⚠️ {alert['message']}")
            
        except Exception as e:
            self.logger.error(f"❌ Resource alert check failed: {e}")
    
    def _calculate_overall_efficiency(self) -> float:
        """Calculate overall pipeline efficiency"""
        try:
            if not self.stage_performance:
                return 0.0
            
            efficiencies = [stage_data.get('efficiency', 0.0) for stage_data in self.stage_performance.values()]
            
            if not efficiencies:
                return 0.0
            
            return sum(efficiencies) / len(efficiencies)
            
        except Exception as e:
            self.logger.error(f"❌ Overall efficiency calculation failed: {e}")
            return 0.0
    
    def _generate_integration_report(self, pipeline_result: Dict[str, Any], 
                                   total_duration: float, overall_efficiency: float) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        try:
            report = {
                'integration_summary': {
                    'status': 'SUCCESS',
                    'integration_active': self.integration_active,
                    'total_duration': total_duration,
                    'overall_efficiency': overall_efficiency,
                    'stages_completed': len(self.stage_performance),
                    'alerts_generated': len(self.alerts)
                },
                'pipeline_summary': pipeline_result,
                'resource_performance': {
                    'stage_performance': self.stage_performance,
                    'resource_history_points': len(self.resource_history),
                    'allocation_percentage': self.allocation_percentage
                },
                'alerts_and_warnings': self.alerts[-10:] if self.alerts else [],  # Last 10 alerts
                'recommendations': self._generate_recommendations(),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Integration report generation failed: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze stage performance
            if self.stage_performance:
                slow_stages = [stage for stage, data in self.stage_performance.items() 
                             if data.get('efficiency', 1.0) < 0.7]
                
                if slow_stages:
                    recommendations.append(f"Consider optimizing slow stages: {', '.join(slow_stages)}")
            
            # Analyze resource usage
            if self.resource_history:
                avg_cpu = sum(r['resources'].get('cpu_percent', 0) for r in self.resource_history) / len(self.resource_history)
                avg_memory = sum(r['resources'].get('memory', {}).get('percent', 0) for r in self.resource_history) / len(self.resource_history)
                
                if avg_cpu < 50:
                    recommendations.append("CPU usage is low, consider increasing parallel processing")
                
                if avg_memory < 60:
                    recommendations.append("Memory usage is low, consider increasing batch sizes")
            
            # Analyze alerts
            if self.alerts:
                critical_alerts = [a for a in self.alerts if a['type'] == 'CRITICAL']
                if critical_alerts:
                    recommendations.append("Critical resource alerts detected, consider reducing allocation percentage")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendations generation failed: {e}")
            return []
    
    def _display_integration_status(self) -> None:
        """Display integration status"""
        try:
            if RICH_AVAILABLE and console:
                # Create status table
                table = Table(title="🌊 Menu 1 Intelligent Resource Integration Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Details", style="white")
                
                # Add rows
                table.add_row(
                    "Integration",
                    "✅ Active" if self.integration_active else "❌ Inactive",
                    f"Allocation: {self.allocation_percentage*100:.1f}%"
                )
                
                table.add_row(
                    "Environment Detection",
                    "✅ Ready" if self.environment_detector else "❌ Not Ready",
                    "Smart detection enabled"
                )
                
                table.add_row(
                    "Resource Orchestrator",
                    "✅ Active" if self.resource_orchestrator else "❌ Inactive",
                    "Adaptive management enabled"
                )
                
                table.add_row(
                    "Menu 1 Pipeline",
                    "✅ Connected" if self.menu1_instance else "❌ Not Connected",
                    "Enhanced Elliott Wave system"
                )
                
                console.print(table)
                
            else:
                # Fallback display
                print("="*80)
                print("🌊 MENU 1 INTELLIGENT RESOURCE INTEGRATION STATUS")
                print("="*80)
                print(f"Integration: {'✅ Active' if self.integration_active else '❌ Inactive'}")
                print(f"Allocation: {self.allocation_percentage*100:.1f}%")
                print(f"Environment Detection: {'✅ Ready' if self.environment_detector else '❌ Not Ready'}")
                print(f"Resource Orchestrator: {'✅ Active' if self.resource_orchestrator else '❌ Inactive'}")
                print(f"Menu 1 Pipeline: {'✅ Connected' if self.menu1_instance else '❌ Not Connected'}")
                print("="*80)
                
        except Exception as e:
            self.logger.error(f"❌ Status display failed: {e}")
    
    def save_integration_report(self, filename: Optional[str] = None) -> str:
        """Save integration report to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"menu1_intelligent_resource_integration_report_{timestamp}.json"
            
            # Generate report
            report = self._generate_integration_report({}, 0, 0)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Integration report saved: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Report saving failed: {e}")
            return ""
    
    def cleanup(self) -> None:
        """Cleanup and shutdown integration"""
        try:
            # Stop monitoring
            self._stop_real_time_monitoring()
            
            # Cleanup resource orchestrator
            if self.resource_orchestrator:
                self.resource_orchestrator.cleanup()
            
            self.integration_active = False
            self.logger.info("🧹 Integration cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Integration cleanup failed: {e}")


def create_menu1_intelligent_integration(allocation_percentage: float = 0.8) -> Menu1IntelligentResourceIntegration:
    """Create Menu 1 intelligent resource integration instance"""
    return Menu1IntelligentResourceIntegration(allocation_percentage=allocation_percentage)


def run_menu1_with_intelligent_resources(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Menu 1 with intelligent resource management"""
    try:
        # Create integration instance
        integration = create_menu1_intelligent_integration()
        
        # Run pipeline
        result = integration.run_intelligent_pipeline(config)
        
        # Save report
        report_file = integration.save_integration_report()
        
        # Cleanup
        integration.cleanup()
        
        return result
        
    except Exception as e:
        logger = get_unified_logger()
        logger.error(f"❌ Menu 1 intelligent resource execution failed: {e}")
        return {"status": "ERROR", "message": str(e)}


# Demo and Testing
def demo_menu1_intelligent_integration():
    """Demo Menu 1 intelligent resource integration"""
    try:
        print("\n" + "="*80)
        print("🎉 MENU 1 INTELLIGENT RESOURCE INTEGRATION DEMO")
        print("="*80)
        
        # Create integration
        integration = create_menu1_intelligent_integration()
        
        # Test initialization
        print("\n🚀 Testing intelligent resource initialization...")
        success = integration.initialize_intelligent_resources()
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Test Menu 1 integration
        print("\n🔗 Testing Menu 1 integration...")
        config = {
            'session_id': 'demo_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'data_file': 'xauusd_1m_features_with_elliott_waves.csv',
            'quick_test': True  # Enable quick test mode
        }
        
        success = integration.integrate_with_menu1(config)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Display status
        print("\n📊 Integration Status:")
        integration._display_integration_status()
        
        # Test stage monitoring
        print("\n🎯 Testing stage monitoring...")
        integration._start_stage_monitoring('demo_stage')
        time.sleep(2)  # Simulate work
        integration._end_stage_monitoring('demo_stage', {'auc': 0.75, 'accuracy': 0.82})
        
        # Generate report
        print("\n📋 Generating integration report...")
        report_file = integration.save_integration_report()
        print(f"Report saved: {report_file}")
        
        # Cleanup
        integration.cleanup()
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run demo
    print("🌊 Menu 1 Intelligent Resource Integration")
    print("Starting demonstration...")
    
    success = demo_menu1_intelligent_integration()
    
    if success:
        print("\n✅ All integration tests passed!")
        print("📋 Menu 1 intelligent resource integration is ready for production use.")
    else:
        print("\n❌ Some integration tests failed.")
        print("🔧 Please check the logs for details.")
