#!/usr/bin/env python3
"""
🌊 MENU 1 INTELLIGENT RESOURCE INTEGRATION
=========================================

การรวมระบบจัดการทรัพยากรอัจฉริยะเข้ากับ Menu 1 Elliott Wave Pipeline
พร้อมการติดตามประสิทธิภาพแบบ real-time และการปรับแต่งอัตโนมัติ

🎯 Features:
- Seamless Menu 1 Integration
- Real-time Resource Monitoring during Pipeline Execution
- Adaptive Resource Allocation
- Performance Analytics & Reporting
- Enterprise-grade Resource Management
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import resource managers
try:
    from core.enhanced_intelligent_resource_manager import EnhancedIntelligentResourceManager, initialize_enhanced_intelligent_resources
except ImportError:
    from enhanced_intelligent_resource_manager import EnhancedIntelligentResourceManager, initialize_enhanced_intelligent_resources

logger = get_unified_logger()

class Menu1ResourceIntegrator:
    """
    🌊 Menu 1 Resource Integration Manager
    ระบบรวมการจัดการทรัพยากรเข้ากับ Menu 1 Elliott Wave Pipeline
    """
    
    def __init__(self, allocation_percentage: float = 0.8):
        """
        Initialize Menu 1 Resource Integrator
        
        Args:
            allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
        """
        self.allocation_percentage = allocation_percentage
        self.resource_manager = None
        self.menu1_instance = None
        self.integration_active = False
        self.stage_callbacks = {}
        self.performance_log = []
        
        logger.info("🌊 Menu 1 Resource Integrator initialized")
    
    def initialize_resource_management(self) -> bool:
        """
        🚀 เริ่มต้นระบบจัดการทรัพยากร
        """
        try:
            self.resource_manager = initialize_enhanced_intelligent_resources(
                allocation_percentage=self.allocation_percentage,
                enable_advanced_monitoring=True
            )
            
            logger.info("🚀 Resource management system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Resource management initialization failed: {e}")
            return False
    
    def integrate_with_menu1(self, menu1_instance) -> bool:
        """
        🔗 รวมระบบเข้ากับ Menu 1 Elliott Wave Pipeline
        
        Args:
            menu1_instance: Instance ของ Menu 1 Elliott Wave
        """
        try:
            if not self.resource_manager:
                self.initialize_resource_management()
            
            # Store Menu 1 reference
            self.menu1_instance = menu1_instance
            
            # Integrate with resource manager
            success = self.resource_manager.integrate_with_menu1(menu1_instance)
            if not success:
                return False
            
            # Setup stage monitoring hooks
            self._setup_menu1_monitoring_hooks()
            
            # Apply optimized configuration
            self._apply_optimized_configuration()
            
            self.integration_active = True
            
            logger.info("🔗 Menu 1 integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Menu 1 integration failed: {e}")
            return False
    
    def _setup_menu1_monitoring_hooks(self) -> None:
        """
        🎣 ตั้งค่า hooks สำหรับการติดตามขั้นตอนต่างๆ ของ Menu 1
        """
        try:
            # Define stage monitoring callbacks
            self.stage_callbacks = {
                'data_loading': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('data_loading'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('data_loading', metrics)
                },
                'feature_engineering': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('feature_engineering'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('feature_engineering', metrics)
                },
                'feature_selection': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('feature_selection'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('feature_selection', metrics)
                },
                'cnn_lstm_training': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('cnn_lstm_training'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('cnn_lstm_training', metrics)
                },
                'dqn_training': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('dqn_training'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('dqn_training', metrics)
                },
                'performance_analysis': {
                    'start': lambda: self.resource_manager.start_stage_monitoring('performance_analysis'),
                    'end': lambda metrics: self.resource_manager.end_stage_monitoring('performance_analysis', metrics)
                }
            }
            
            logger.info("🎣 Menu 1 monitoring hooks setup completed")
            
        except Exception as e:
            logger.error(f"❌ Monitoring hooks setup failed: {e}")
    
    def _apply_optimized_configuration(self) -> None:
        """
        🔧 ประยุกต์ configuration ที่ปรับแต่งแล้วกับ Menu 1
        """
        try:
            if not self.menu1_instance or not self.resource_manager:
                return
            
            # Get optimized configuration
            optimized_config = self.resource_manager.get_menu1_optimization_config()
            
            # Apply to Menu 1 if it has config attribute
            if hasattr(self.menu1_instance, 'config'):
                # Merge configurations safely
                for section, section_config in optimized_config.items():
                    if section not in self.menu1_instance.config:
                        self.menu1_instance.config[section] = {}
                    
                    self.menu1_instance.config[section].update(section_config)
                
                logger.info("🔧 Optimized configuration applied to Menu 1")
            else:
                logger.warning("⚠️ Menu 1 instance has no config attribute")
                
        except Exception as e:
            logger.error(f"❌ Configuration application failed: {e}")
    
    def start_stage(self, stage_name: str) -> None:
        """
        ▶️ เริ่มขั้นตอนและเริ่มติดตาม
        
        Args:
            stage_name: ชื่อขั้นตอน
        """
        try:
            if not self.integration_active or not self.resource_manager:
                return
            
            if stage_name in self.stage_callbacks:
                callback = self.stage_callbacks[stage_name]['start']
                callback()
                
                # Log stage start
                self.performance_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'stage_start',
                    'stage': stage_name
                })
                
                logger.info(f"▶️ Started monitoring stage: {stage_name}")
            
        except Exception as e:
            logger.error(f"❌ Stage start failed for {stage_name}: {e}")
    
    def end_stage(self, stage_name: str, performance_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ⏹️ สิ้นสุดขั้นตอนและได้รับสรุปประสิทธิภาพ
        
        Args:
            stage_name: ชื่อขั้นตอน
            performance_metrics: เมตริกส์ประสิทธิภาพจากขั้นตอน
        
        Returns:
            Dict: สรุปประสิทธิภาพของขั้นตอน
        """
        try:
            if not self.integration_active or not self.resource_manager:
                return {}
            
            summary = {}
            if stage_name in self.stage_callbacks:
                callback = self.stage_callbacks[stage_name]['end']
                summary = callback(performance_metrics or {})
                
                # Check for adaptive adjustments
                adjustments = self.resource_manager.adaptive_resource_adjustment(
                    stage_name, performance_metrics or {}
                )
                
                if adjustments:
                    logger.info(f"🔄 Adaptive adjustments for {stage_name}: {adjustments}")
                    summary['adaptive_adjustments'] = adjustments
                
                # Log stage end
                self.performance_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'stage_end',
                    'stage': stage_name,
                    'summary': summary
                })
                
                logger.info(f"⏹️ Completed monitoring stage: {stage_name}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Stage end failed for {stage_name}: {e}")
            return {}
    
    def get_current_performance_status(self) -> Dict[str, Any]:
        """
        📊 รับสถานะประสิทธิภาพปัจจุบัน
        """
        try:
            if not self.resource_manager:
                return {}
            
            current_perf = self.resource_manager.get_current_performance()
            alerts = self.resource_manager.get_performance_alerts(5)
            
            status = {
                'resource_utilization': {
                    'cpu_percent': current_perf.get('cpu_percent', 0),
                    'memory_percent': current_perf.get('memory', {}).get('percent', 0),
                    'uptime_minutes': current_perf.get('uptime_minutes', 0)
                },
                'performance_alerts': alerts,
                'integration_status': {
                    'active': self.integration_active,
                    'monitoring_stages': len(self.stage_callbacks),
                    'performance_log_entries': len(self.performance_log)
                },
                'resource_allocation': self.resource_manager.resource_config.get('optimization', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Performance status retrieval failed: {e}")
            return {}
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """
        📋 สร้างรายงานการรวมระบบ
        """
        try:
            if not self.resource_manager:
                return {}
            
            # Get pipeline performance report
            pipeline_report = self.resource_manager.generate_pipeline_performance_report()
            
            # Add integration-specific information
            integration_report = {
                **pipeline_report,
                'integration_details': {
                    'menu1_integration_active': self.integration_active,
                    'resource_allocation_percentage': self.allocation_percentage,
                    'monitoring_hooks_setup': len(self.stage_callbacks),
                    'performance_log_entries': len(self.performance_log)
                },
                'menu1_specific_metrics': {
                    'optimized_configuration_applied': hasattr(self.menu1_instance, 'config') if self.menu1_instance else False,
                    'adaptive_adjustments_count': len([log for log in self.performance_log if 'adaptive_adjustments' in log.get('summary', {})]),
                    'stage_completion_rate': self._calculate_stage_completion_rate()
                },
                'performance_timeline': self.performance_log[-20:] if self.performance_log else [],  # Last 20 events
                'integration_timestamp': datetime.now().isoformat()
            }
            
            return integration_report
            
        except Exception as e:
            logger.error(f"❌ Integration report generation failed: {e}")
            return {}
    
    def _calculate_stage_completion_rate(self) -> float:
        """
        📈 คำนวณอัตราการสำเร็จของขั้นตอน
        """
        try:
            start_events = [log for log in self.performance_log if log.get('event') == 'stage_start']
            end_events = [log for log in self.performance_log if log.get('event') == 'stage_end']
            
            if not start_events:
                return 0.0
            
            completion_rate = len(end_events) / len(start_events)
            return round(completion_rate, 3)
            
        except Exception:
            return 0.0
    
    def display_integration_dashboard(self) -> None:
        """
        📊 แสดง dashboard การรวมระบบ
        """
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*80)
            print("🌊 MENU 1 INTELLIGENT RESOURCE INTEGRATION DASHBOARD")
            print("="*80)
            
            # Integration status
            print(f"\n🔗 INTEGRATION STATUS:")
            print(f"   Active: {'✅ Yes' if self.integration_active else '❌ No'}")
            print(f"   Menu 1 Instance: {'✅ Connected' if self.menu1_instance else '❌ Not Connected'}")
            print(f"   Resource Manager: {'✅ Ready' if self.resource_manager else '❌ Not Ready'}")
            
            # Current performance
            if self.resource_manager:
                current_status = self.get_current_performance_status()
                resource_util = current_status.get('resource_utilization', {})
                
                print(f"\n📊 CURRENT PERFORMANCE:")
                print(f"   CPU: {resource_util.get('cpu_percent', 0):.1f}%")
                print(f"   Memory: {resource_util.get('memory_percent', 0):.1f}%")
                print(f"   Uptime: {resource_util.get('uptime_minutes', 0):.1f} minutes")
                
                # Performance alerts
                alerts = current_status.get('performance_alerts', [])
                if alerts:
                    print(f"\n🚨 RECENT ALERTS:")
                    for alert in alerts[-3:]:
                        print(f"   {alert.get('type', 'INFO').upper()}: {alert.get('message', 'N/A')}")
                
                # Resource allocation
                allocation = current_status.get('resource_allocation', {})
                print(f"\n⚡ RESOURCE ALLOCATION:")
                print(f"   Batch Size: {allocation.get('batch_size', 'N/A')}")
                print(f"   Workers: {allocation.get('recommended_workers', 'N/A')}")
                print(f"   Memory Limit: {allocation.get('memory_limit_gb', 'N/A')} GB")
            
            # Recent activity
            if self.performance_log:
                print(f"\n📝 RECENT ACTIVITY:")
                for log_entry in self.performance_log[-5:]:
                    event = log_entry.get('event', 'unknown')
                    stage = log_entry.get('stage', 'unknown')
                    timestamp = log_entry.get('timestamp', '')[:19]  # Remove microseconds
                    print(f"   {timestamp}: {event} - {stage}")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Integration dashboard display failed: {e}")
    
    def save_integration_report(self, file_path: Optional[str] = None) -> str:
        """
        💾 บันทึกรายงานการรวมระบบ
        """
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"menu1_integration_report_{timestamp}.json"
            
            report = self.generate_integration_report()
            
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Integration report saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Integration report saving failed: {e}")
            return ""
    
    def cleanup(self) -> None:
        """
        🧹 ทำความสะอาดและปิดระบบ
        """
        try:
            if self.resource_manager:
                self.resource_manager.stop_monitoring()
            
            self.integration_active = False
            logger.info("🧹 Integration cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Integration cleanup failed: {e}")


def create_menu1_resource_integrator(allocation_percentage: float = 0.8) -> Menu1ResourceIntegrator:
    """
    🚀 สร้าง Menu 1 Resource Integrator
    
    Args:
        allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
        
    Returns:
        Menu1ResourceIntegrator: ตัวรวมระบบที่พร้อมใช้งาน
    """
    try:
        integrator = Menu1ResourceIntegrator(allocation_percentage)
        success = integrator.initialize_resource_management()
        
        if not success:
            raise Exception("Resource management initialization failed")
        
        logger.info("🚀 Menu 1 Resource Integrator created successfully")
        return integrator
        
    except Exception as e:
        logger.error(f"❌ Menu 1 Resource Integrator creation failed: {e}")
        raise


# Example usage with Mock Menu 1 for testing
if __name__ == "__main__":
    try:
        print("🌊 Testing Menu 1 Intelligent Resource Integration...")
        
        # Create mock Menu 1 instance
        class MockMenu1:
            def __init__(self):
                self.config = {}
        
        mock_menu1 = MockMenu1()
        
        # Create integrator
        integrator = create_menu1_resource_integrator()
        
        # Integrate with mock Menu 1
        success = integrator.integrate_with_menu1(mock_menu1)
        if not success:
            raise Exception("Integration failed")
        
        print("✅ Integration successful!")
        
        # Test stage monitoring
        stages = ['data_loading', 'feature_engineering', 'cnn_lstm_training']
        
        for stage in stages:
            print(f"\n▶️ Testing {stage}...")
            integrator.start_stage(stage)
            
            time.sleep(1)  # Simulate work
            
            # Mock performance metrics
            metrics = {
                'auc': 0.75,
                'accuracy': 0.82,
                'duration': 45.5
            }
            
            summary = integrator.end_stage(stage, metrics)
            print(f"⏹️ {stage} completed - Efficiency: {summary.get('efficiency_score', 0):.2f}")
        
        # Display dashboard
        integrator.display_integration_dashboard()
        
        # Generate and save report
        report_file = integrator.save_integration_report()
        print(f"\n📋 Integration report saved: {report_file}")
        
        # Cleanup
        integrator.cleanup()
        
        print("\n✅ Menu 1 Intelligent Resource Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

        traceback.print_exc()
