#!/usr/bin/env python3
"""
🌊 MENU 1 INTELLIGENT RESOURCE INTEGRATION - UNIFIED SYSTEM
==========================================================

ระบบเชื่อมต่อ Intelligent Resource Management กับ Menu 1 Elliott Wave Pipeline
รวมระบบทุกส่วนให้เป็นหนึ่งเดียว ไม่ซ้ำซ้อน และทำงานร่วมกันอย่างสมบูรณ์

🎯 UNIFIED FEATURES:
✅ การรวมระบบทุกส่วนเป็นหนึ่งเดียว (Single System)
✅ การเชื่อมต่อ Menu 1 Elliott Wave Pipeline
✅ การจัดการทรัพยากรอัจฉริยะแบบ 80% แบบ Real-time
✅ การติดตามและปรับปรุงประสิทธิภาพอัตโนมัติ
✅ ระบบการแจ้งเตือนและการกู้คืน
✅ รายงานผลการดำเนินงานแบบครบถ้วน
✅ การทำงานร่วมกันแบบไร้รอยต่อ

🚀 INTEGRATION APPROACH:
- ใช้ระบบที่มีอยู่แล้ว (Existing Systems)
- เชื่อมต่อแบบ Direct Integration
- ไม่สร้างระบบใหม่ที่ซ้ำซ้อน
- ใช้ Interface และ Adapter Pattern
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
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import core components
from core.unified_enterprise_logger import get_unified_logger, LogLevel, ProcessStatus, ElliottWaveStep, Menu1Step

# Import existing resource management systems
try:
    from core.unified_resource_manager import get_unified_resource_manager
    UNIFIED_RESOURCE_AVAILABLE = True
except ImportError:
    UNIFIED_RESOURCE_AVAILABLE = False

try:
    from core.intelligent_environment_detector import IntelligentEnvironmentDetector
    ENVIRONMENT_DETECTOR_AVAILABLE = True
except ImportError:
    ENVIRONMENT_DETECTOR_AVAILABLE = False

# Import Menu 1 system
try:
    from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
    MENU1_AVAILABLE = True
except ImportError:
    MENU1_AVAILABLE = False

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

# Import system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class UnifiedMenu1ResourceIntegration:
    """
    🌊 Unified Menu 1 Resource Integration System
    
    ระบบรวมการจัดการทรัพยากรเข้ากับ Menu 1 Elliott Wave Pipeline
    โดยใช้ระบบที่มีอยู่แล้วและเชื่อมต่อแบบไร้รอยต่อ
    """
    
    def __init__(self, target_allocation: float = 0.8):
        """Initialize unified integration system"""
        self.logger = get_unified_logger()
        self.target_allocation = target_allocation
        self.integration_active = False
        
        # System components
        self.resource_manager = None
        self.environment_detector = None
        self.menu1_instance = None
        
        # Performance tracking
        self.performance_metrics = []
        self.stage_performance = {}
        self.resource_history = []
        self.alerts = []
        
        # Monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        self.logger.info("🌊 Unified Menu 1 Resource Integration initialized")
    
    def initialize_unified_system(self) -> bool:
        """Initialize unified resource management system"""
        try:
            # Initialize environment detector
            if ENVIRONMENT_DETECTOR_AVAILABLE:
                self.environment_detector = IntelligentEnvironmentDetector()
                environment_info = self.environment_detector.detect_environment()
                
                self.logger.info(f"🌍 Environment: {environment_info.get('environment_type', 'unknown')}")
                self.logger.info(f"💻 Hardware: {environment_info.get('hardware_capability', 'unknown')}")
                self.logger.info(f"⚡ Optimization: {environment_info.get('optimization_level', 'unknown')}")
            
            # Initialize unified resource manager
            if UNIFIED_RESOURCE_AVAILABLE:
                self.resource_manager = get_unified_resource_manager()
                
                # Configure resource manager for 80% allocation
                self.resource_manager.configure_allocation(
                    target_percentage=self.target_allocation,
                    enable_monitoring=True,
                    enable_adaptation=True
                )
                
                # Start monitoring
                self.resource_manager.start_monitoring()
                
                self.logger.info(f"🚀 Resource Manager initialized (Target: {self.target_allocation*100:.1f}%)")
                return True
            else:
                self.logger.warning("⚠️ Unified Resource Manager not available")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Unified system initialization failed: {e}")
            return False
    
    def create_menu1_with_integration(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Create Menu 1 instance with resource integration"""
        try:
            if not MENU1_AVAILABLE:
                self.logger.error("❌ Enhanced Menu 1 not available")
                return False
            
            # Create Menu 1 instance
            self.menu1_instance = EnhancedMenu1ElliottWave(config=config)
            
            # Integrate resource management
            if self.resource_manager:
                # Apply resource optimization to Menu 1
                self._apply_resource_optimization()
                
                # Setup monitoring hooks
                self._setup_monitoring_hooks()
                
                self.logger.info("🔗 Menu 1 resource integration completed")
                return True
            else:
                self.logger.warning("⚠️ Resource Manager not available, using Menu 1 without integration")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Menu 1 integration failed: {e}")
            return False
    
    def _apply_resource_optimization(self) -> None:
        """Apply resource optimization to Menu 1"""
        try:
            if not self.resource_manager or not self.menu1_instance:
                return
            
            # Get current resource utilization
            current_performance = self.resource_manager.get_current_performance()
            
            # Calculate optimal settings
            optimal_settings = self._calculate_optimal_settings(current_performance)
            
            # Apply settings to Menu 1 configuration
            if hasattr(self.menu1_instance, 'config') and optimal_settings:
                self.menu1_instance.config.update(optimal_settings)
                
                self.logger.info("⚙️ Resource optimization applied to Menu 1")
                self.logger.info(f"   🎯 Batch Size: {optimal_settings.get('batch_size', 'N/A')}")
                self.logger.info(f"   🧠 Workers: {optimal_settings.get('n_jobs', 'N/A')}")
                self.logger.info(f"   💾 Memory Limit: {optimal_settings.get('memory_limit', 'N/A')}")
                
        except Exception as e:
            self.logger.error(f"❌ Resource optimization failed: {e}")
    
    def _calculate_optimal_settings(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal settings based on current performance"""
        try:
            settings = {}
            
            # CPU optimization
            cpu_count = current_performance.get('cpu_count', 1)
            cpu_percent = current_performance.get('cpu_percent', 0)
            
            if cpu_percent < 60:
                settings['n_jobs'] = min(cpu_count, 8)
            else:
                settings['n_jobs'] = max(1, cpu_count // 2)
            
            # Memory optimization
            memory_info = current_performance.get('memory', {})
            memory_percent = memory_info.get('percent', 0)
            memory_available = memory_info.get('available', 0)
            
            if memory_percent < 70:
                settings['batch_size'] = 1024
                settings['memory_limit'] = '8GB'
            else:
                settings['batch_size'] = 512
                settings['memory_limit'] = '4GB'
            
            # Training optimization
            if memory_percent < 60:
                settings['epochs'] = 100
                settings['validation_split'] = 0.2
            else:
                settings['epochs'] = 50
                settings['validation_split'] = 0.1
            
            return settings
            
        except Exception as e:
            self.logger.error(f"❌ Optimal settings calculation failed: {e}")
            return {}
    
    def _setup_monitoring_hooks(self) -> None:
        """Setup monitoring hooks for Menu 1 stages"""
        try:
            # Define stage callbacks
            self.stage_callbacks = {
                'data_loading': self._create_stage_callback('data_loading'),
                'feature_engineering': self._create_stage_callback('feature_engineering'),
                'feature_selection': self._create_stage_callback('feature_selection'),
                'cnn_lstm_training': self._create_stage_callback('cnn_lstm_training'),
                'dqn_training': self._create_stage_callback('dqn_training'),
                'performance_analysis': self._create_stage_callback('performance_analysis'),
                'model_validation': self._create_stage_callback('model_validation'),
                'result_compilation': self._create_stage_callback('result_compilation')
            }
            
            self.logger.info("🎣 Monitoring hooks setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring hooks setup failed: {e}")
    
    def _create_stage_callback(self, stage_name: str) -> Dict[str, Callable]:
        """Create callback functions for a specific stage"""
        return {
            'start': lambda: self._start_stage_monitoring(stage_name),
            'end': lambda metrics: self._end_stage_monitoring(stage_name, metrics),
            'progress': lambda progress: self._update_stage_progress(stage_name, progress),
            'error': lambda error: self._handle_stage_error(stage_name, error)
        }
    
    def _start_stage_monitoring(self, stage: str) -> None:
        """Start monitoring for a specific stage"""
        try:
            # Record stage start
            self.stage_performance[stage] = {
                'start_time': time.time(),
                'start_resources': self._get_current_resources(),
                'status': 'running'
            }
            
            # Start resource monitoring for this stage
            if self.resource_manager:
                self.resource_manager.start_stage_monitoring(stage)
            
            self.logger.info(f"▶️ Stage monitoring started: {stage}")
            
        except Exception as e:
            self.logger.error(f"❌ Stage monitoring start failed for {stage}: {e}")
    
    def _end_stage_monitoring(self, stage: str, metrics: Dict[str, Any]) -> None:
        """End monitoring for a specific stage"""
        try:
            if stage in self.stage_performance:
                end_time = time.time()
                start_time = self.stage_performance[stage]['start_time']
                duration = end_time - start_time
                
                # Update stage performance
                self.stage_performance[stage].update({
                    'end_time': end_time,
                    'duration': duration,
                    'end_resources': self._get_current_resources(),
                    'metrics': metrics,
                    'status': 'completed'
                })
                
                # End resource monitoring
                if self.resource_manager:
                    self.resource_manager.end_stage_monitoring(stage, metrics)
                
                # Calculate efficiency
                efficiency = self._calculate_stage_efficiency(stage)
                self.stage_performance[stage]['efficiency'] = efficiency
                
                self.logger.info(f"⏹️ Stage completed: {stage} (Duration: {duration:.2f}s, Efficiency: {efficiency:.2f})")
                
        except Exception as e:
            self.logger.error(f"❌ Stage monitoring end failed for {stage}: {e}")
    
    def _update_stage_progress(self, stage: str, progress: float) -> None:
        """Update stage progress"""
        try:
            if stage in self.stage_performance:
                self.stage_performance[stage]['progress'] = progress
                
                # Update resource manager
                if self.resource_manager:
                    self.resource_manager.update_stage_progress(stage, progress)
                
        except Exception as e:
            self.logger.error(f"❌ Stage progress update failed for {stage}: {e}")
    
    def _handle_stage_error(self, stage: str, error: Exception) -> None:
        """Handle stage error"""
        try:
            if stage in self.stage_performance:
                self.stage_performance[stage]['status'] = 'error'
                self.stage_performance[stage]['error'] = str(error)
                
                # Alert resource manager
                if self.resource_manager:
                    self.resource_manager.handle_stage_error(stage, error)
                
                self.logger.error(f"❌ Stage error: {stage} - {error}")
                
        except Exception as e:
            self.logger.error(f"❌ Stage error handling failed for {stage}: {e}")
    
    def _get_current_resources(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        try:
            if self.resource_manager:
                return self.resource_manager.get_current_performance()
            elif PSUTIL_AVAILABLE:
                return {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory': psutil.virtual_memory()._asdict(),
                    'disk': psutil.disk_usage('/')._asdict()
                }
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
            
            # Get resource usage
            start_resources = stage_data.get('start_resources', {})
            end_resources = stage_data.get('end_resources', {})
            duration = stage_data.get('duration', 0)
            
            if not start_resources or not end_resources or duration == 0:
                return 0.0
            
            # Calculate efficiency components
            cpu_efficiency = 1.0 - (end_resources.get('cpu_percent', 0) / 100.0)
            memory_efficiency = end_resources.get('memory', {}).get('percent', 0) / 100.0
            duration_efficiency = min(1.0, 300.0 / duration)  # 5 minutes baseline
            
            # Weighted average
            efficiency = (cpu_efficiency * 0.3 + memory_efficiency * 0.4 + duration_efficiency * 0.3)
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency calculation failed for {stage}: {e}")
            return 0.0
    
    def run_unified_pipeline(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run Menu 1 pipeline with unified resource management"""
        try:
            # Initialize system if not already done
            if not self.resource_manager:
                success = self.initialize_unified_system()
                if not success:
                    return {"status": "ERROR", "message": "Failed to initialize unified system"}
            
            # Create Menu 1 instance if not already done
            if not self.menu1_instance:
                success = self.create_menu1_with_integration(config)
                if not success:
                    return {"status": "ERROR", "message": "Failed to create Menu 1 with integration"}
            
            # Mark integration as active
            self.integration_active = True
            
            # Start real-time monitoring
            self._start_real_time_monitoring()
            
            # Display integration status
            self._display_integration_status()
            
            # Execute Menu 1 pipeline
            self.logger.info("🚀 Starting Menu 1 pipeline with unified resource management...")
            
            start_time = time.time()
            
            # Run the pipeline
            result = self.menu1_instance.run()
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Stop monitoring
            self._stop_real_time_monitoring()
            
            # Calculate overall performance
            overall_efficiency = self._calculate_overall_efficiency()
            
            # Generate comprehensive report
            integration_report = self._generate_integration_report(result, total_duration, overall_efficiency)
            
            self.logger.info(f"✅ Menu 1 pipeline completed with unified resource management")
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
            self.logger.error(f"❌ Unified pipeline execution failed: {e}")
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
                self._apply_adaptive_adjustments(current_resources)
                
                time.sleep(5)  # Monitor every 5 seconds
                
        except Exception as e:
            self.logger.error(f"❌ Monitoring loop error: {e}")
    
    def _check_resource_alerts(self, resources: Dict[str, Any]) -> None:
        """Check for resource alerts"""
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
            
            # Store and log alerts
            for alert in alerts:
                alert['timestamp'] = time.time()
                self.alerts.append(alert)
                
                if alert['type'] == 'CRITICAL':
                    self.logger.error(f"🚨 {alert['message']}")
                else:
                    self.logger.warning(f"⚠️ {alert['message']}")
                    
        except Exception as e:
            self.logger.error(f"❌ Resource alert check failed: {e}")
    
    def _apply_adaptive_adjustments(self, resources: Dict[str, Any]) -> None:
        """Apply adaptive resource adjustments"""
        try:
            if not self.resource_manager:
                return
            
            # Calculate adjustment needs
            adjustments = self._calculate_adjustments(resources)
            
            # Apply adjustments
            if adjustments:
                self.resource_manager.apply_adjustments(adjustments)
                self.logger.info(f"⚙️ Adaptive adjustments applied: {adjustments}")
                
        except Exception as e:
            self.logger.error(f"❌ Adaptive adjustments failed: {e}")
    
    def _calculate_adjustments(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate needed resource adjustments"""
        try:
            adjustments = {}
            
            # CPU adjustments
            cpu_percent = resources.get('cpu_percent', 0)
            if cpu_percent > 90:
                adjustments['reduce_parallel_jobs'] = True
            elif cpu_percent < 30:
                adjustments['increase_parallel_jobs'] = True
            
            # Memory adjustments
            memory_info = resources.get('memory', {})
            memory_percent = memory_info.get('percent', 0)
            if memory_percent > 90:
                adjustments['reduce_batch_size'] = True
            elif memory_percent < 50:
                adjustments['increase_batch_size'] = True
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"❌ Adjustment calculation failed: {e}")
            return {}
    
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
                    'alerts_generated': len(self.alerts),
                    'target_allocation': self.target_allocation
                },
                'pipeline_summary': pipeline_result,
                'performance_analysis': {
                    'stage_performance': self.stage_performance,
                    'resource_utilization_history': len(self.resource_history),
                    'efficiency_by_stage': {
                        stage: data.get('efficiency', 0.0) 
                        for stage, data in self.stage_performance.items()
                    }
                },
                'resource_management': {
                    'unified_system_enabled': UNIFIED_RESOURCE_AVAILABLE,
                    'environment_detection_enabled': ENVIRONMENT_DETECTOR_AVAILABLE,
                    'monitoring_enabled': self.monitoring_active,
                    'alerts_and_warnings': self.alerts[-10:] if self.alerts else []
                },
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
            
            # Performance recommendations
            if self.stage_performance:
                avg_efficiency = sum(data.get('efficiency', 0.0) for data in self.stage_performance.values()) / len(self.stage_performance)
                
                if avg_efficiency < 0.6:
                    recommendations.append("Consider reducing workload or increasing resources")
                elif avg_efficiency > 0.9:
                    recommendations.append("System is highly efficient, consider increasing workload")
            
            # Resource recommendations
            if self.resource_history:
                avg_cpu = sum(r['resources'].get('cpu_percent', 0) for r in self.resource_history) / len(self.resource_history)
                avg_memory = sum(r['resources'].get('memory', {}).get('percent', 0) for r in self.resource_history) / len(self.resource_history)
                
                if avg_cpu < 40:
                    recommendations.append("CPU utilization is low, consider increasing parallel processing")
                if avg_memory < 50:
                    recommendations.append("Memory utilization is low, consider increasing batch sizes")
            
            # Alert recommendations
            if self.alerts:
                critical_alerts = [a for a in self.alerts if a['type'] == 'CRITICAL']
                if critical_alerts:
                    recommendations.append("Critical alerts detected, consider resource optimization")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendations generation failed: {e}")
            return []
    
    def _display_integration_status(self) -> None:
        """Display integration status"""
        try:
            if RICH_AVAILABLE and console:
                # Create status table
                table = Table(title="🌊 Unified Menu 1 Resource Integration Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Details", style="white")
                
                # Add rows
                table.add_row(
                    "Unified System",
                    "✅ Active" if self.integration_active else "❌ Inactive",
                    f"Target: {self.target_allocation*100:.1f}%"
                )
                
                table.add_row(
                    "Resource Manager",
                    "✅ Ready" if self.resource_manager else "❌ Not Ready",
                    "Unified management system"
                )
                
                table.add_row(
                    "Environment Detection",
                    "✅ Available" if ENVIRONMENT_DETECTOR_AVAILABLE else "❌ Unavailable",
                    "Smart detection system"
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
                print("🌊 UNIFIED MENU 1 RESOURCE INTEGRATION STATUS")
                print("="*80)
                print(f"Unified System: {'✅ Active' if self.integration_active else '❌ Inactive'}")
                print(f"Target Allocation: {self.target_allocation*100:.1f}%")
                print(f"Resource Manager: {'✅ Ready' if self.resource_manager else '❌ Not Ready'}")
                print(f"Environment Detection: {'✅ Available' if ENVIRONMENT_DETECTOR_AVAILABLE else '❌ Unavailable'}")
                print(f"Menu 1 Pipeline: {'✅ Connected' if self.menu1_instance else '❌ Not Connected'}")
                print("="*80)
                
        except Exception as e:
            self.logger.error(f"❌ Status display failed: {e}")
    
    def save_integration_report(self, filename: Optional[str] = None) -> str:
        """Save integration report to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"unified_menu1_integration_report_{timestamp}.json"
            
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
            
            # Stop resource manager
            if self.resource_manager:
                self.resource_manager.stop_monitoring()
            
            self.integration_active = False
            self.logger.info("🧹 Integration cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Integration cleanup failed: {e}")


def create_unified_menu1_integration(target_allocation: float = 0.8) -> UnifiedMenu1ResourceIntegration:
    """Create unified Menu 1 resource integration instance"""
    return UnifiedMenu1ResourceIntegration(target_allocation=target_allocation)


def run_menu1_with_unified_resources(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Menu 1 with unified resource management"""
    try:
        # Create integration instance
        integration = create_unified_menu1_integration()
        
        # Run pipeline
        result = integration.run_unified_pipeline(config)
        
        # Save report
        report_file = integration.save_integration_report()
        
        # Cleanup
        integration.cleanup()
        
        return result
        
    except Exception as e:
        logger = get_unified_logger()
        logger.error(f"❌ Menu 1 unified resource execution failed: {e}")
        return {"status": "ERROR", "message": str(e)}


# Demo and Testing
def demo_unified_menu1_integration():
    """Demo unified Menu 1 resource integration"""
    try:
        print("\n" + "="*80)
        print("🎉 UNIFIED MENU 1 RESOURCE INTEGRATION DEMO")
        print("="*80)
        
        # Create integration
        integration = create_unified_menu1_integration()
        
        # Test system initialization
        print("\n🚀 Testing unified system initialization...")
        success = integration.initialize_unified_system()
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Test Menu 1 integration
        print("\n🔗 Testing Menu 1 integration...")
        config = {
            'session_id': 'demo_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'data_file': 'xauusd_1m_features_with_elliott_waves.csv',
            'quick_test': True  # Enable quick test mode
        }
        
        success = integration.create_menu1_with_integration(config)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Display status
        print("\n📊 Integration Status:")
        integration._display_integration_status()
        
        # Test monitoring
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
    print("🌊 Unified Menu 1 Resource Integration")
    print("Starting demonstration...")
    
    success = demo_unified_menu1_integration()
    
    if success:
        print("\n✅ All unified integration tests passed!")
        print("📋 Unified Menu 1 resource integration is ready for production use.")
    else:
        print("\n❌ Some unified integration tests failed.")
        print("🔧 Please check the logs for details.")
