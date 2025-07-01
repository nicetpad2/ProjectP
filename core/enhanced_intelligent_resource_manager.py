#!/usr/bin/env python3
"""
⚡ ENHANCED INTELLIGENT RESOURCE MANAGER
========================================

Enhanced version ของระบบจัดการทรัพยากรอัจฉริยะ สำหรับการรวมเข้ากับ Menu 1 Elliott Wave Pipeline
พร้อมด้วยการติดตามแบบ real-time และการปรับแต่งอัตโนมัติ

🎯 Advanced Features:
- Deep Menu 1 Integration
- Real-time Performance Analytics
- Adaptive Resource Allocation
- Enterprise-grade Monitoring
- Automatic Performance Tuning
"""

import os
import sys
import numpy as np
import pandas as pd
import threading
import queue
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Import base resource manager
try:
    from .intelligent_resource_manager import IntelligentResourceManager
except ImportError:
    from intelligent_resource_manager import IntelligentResourceManager

logger = logging.getLogger(__name__)

class EnhancedIntelligentResourceManager(IntelligentResourceManager):
    """
    ⚡ Enhanced Intelligent Resource Manager
    ระบบจัดการทรัพยากรขั้นสูงพร้อม Menu 1 Integration
    """
    
    def __init__(self, allocation_percentage: float = 0.8, enable_advanced_monitoring: bool = True):
        """
        Initialize Enhanced Resource Manager
        
        Args:
            allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
            enable_advanced_monitoring: เปิดใช้งานการติดตามขั้นสูง
        """
        super().__init__(allocation_percentage)
        
        self.enable_advanced_monitoring = enable_advanced_monitoring
        self.pipeline_stage_monitoring = {}
        self.performance_alerts = queue.Queue()
        self.optimization_history = []
        self.menu1_integration_active = False
        
        # Advanced monitoring metrics
        self.stage_performance = {}
        self.resource_efficiency_scores = []
        self.adaptive_adjustments = []
        
        # Menu 1 specific configurations
        self.menu1_stages = [
            'data_loading',
            'feature_engineering', 
            'feature_selection',
            'cnn_lstm_training',
            'dqn_training',
            'performance_analysis'
        ]
        
        logger.info("⚡ Enhanced Intelligent Resource Manager initialized")
    
    def integrate_with_menu1(self, menu1_instance) -> bool:
        """
        🌊 รวมระบบเข้ากับ Menu 1 Elliott Wave Pipeline
        
        Args:
            menu1_instance: Instance ของ Menu 1 Elliott Wave
            
        Returns:
            bool: สำเร็จหรือไม่
        """
        try:
            # Store Menu 1 reference
            self.menu1_instance = menu1_instance
            self.menu1_integration_active = True
            
            # Apply Menu 1 optimized configuration
            menu1_config = self.get_menu1_optimization_config()
            self._apply_menu1_configuration(menu1_config)
            
            # Setup stage monitoring
            self._setup_pipeline_stage_monitoring()
            
            # Start advanced monitoring
            if self.enable_advanced_monitoring:
                self._start_advanced_monitoring()
            
            logger.info("🌊 Menu 1 integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Menu 1 integration failed: {e}")
            return False
    
    def _apply_menu1_configuration(self, config: Dict[str, Any]) -> None:
        """
        🔧 ประยุกต์ configuration ที่เหมาะสมกับ Menu 1
        """
        try:
            # Update Menu 1 instance configuration if available
            if hasattr(self, 'menu1_instance') and self.menu1_instance:
                # Apply data processing configuration
                data_config = config.get('data_processing', {})
                if hasattr(self.menu1_instance, 'config'):
                    if 'data_processing' not in self.menu1_instance.config:
                        self.menu1_instance.config['data_processing'] = {}
                    self.menu1_instance.config['data_processing'].update(data_config)
                
                # Apply ML framework configurations
                for framework in ['elliott_wave', 'feature_selection', 'cnn_lstm', 'dqn']:
                    framework_config = config.get(framework, {})
                    if framework_config:
                        if framework not in self.menu1_instance.config:
                            self.menu1_instance.config[framework] = {}
                        self.menu1_instance.config[framework].update(framework_config)
            
            logger.info("🔧 Menu 1 configuration applied successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Menu 1 configuration application partial: {e}")
    
    def _setup_pipeline_stage_monitoring(self) -> None:
        """
        📊 ตั้งค่าการติดตามแต่ละขั้นตอนของ pipeline
        """
        try:
            for stage in self.menu1_stages:
                self.pipeline_stage_monitoring[stage] = {
                    'start_time': None,
                    'end_time': None,
                    'resource_usage': [],
                    'performance_metrics': {},
                    'efficiency_score': 0.0
                }
            
            logger.info("📊 Pipeline stage monitoring setup completed")
            
        except Exception as e:
            logger.error(f"❌ Pipeline monitoring setup failed: {e}")
    
    def start_stage_monitoring(self, stage_name: str) -> None:
        """
        ▶️ เริ่มติดตามขั้นตอนเฉพาะ
        
        Args:
            stage_name: ชื่อขั้นตอนที่ต้องการติดตาม
        """
        try:
            if stage_name in self.pipeline_stage_monitoring:
                stage_info = self.pipeline_stage_monitoring[stage_name]
                stage_info['start_time'] = datetime.now()
                stage_info['resource_usage'] = []
                
                # Collect initial resource state
                initial_state = self._collect_current_resource_state()
                stage_info['resource_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'stage_start',
                    **initial_state
                })
                
                logger.info(f"▶️ Started monitoring stage: {stage_name}")
            
        except Exception as e:
            logger.error(f"❌ Stage monitoring start failed for {stage_name}: {e}")
    
    def end_stage_monitoring(self, stage_name: str, performance_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ⏹️ สิ้นสุดการติดตามขั้นตอนและคำนวณประสิทธิภาพ
        
        Args:
            stage_name: ชื่อขั้นตอน
            performance_metrics: เมตริกส์ประสิทธิภาพจากขั้นตอน
            
        Returns:
            Dict: สรุปประสิทธิภาพของขั้นตอน
        """
        try:
            if stage_name not in self.pipeline_stage_monitoring:
                return {}
            
            stage_info = self.pipeline_stage_monitoring[stage_name]
            stage_info['end_time'] = datetime.now()
            
            # Collect final resource state
            final_state = self._collect_current_resource_state()
            stage_info['resource_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'stage_end',
                **final_state
            })
            
            # Store performance metrics
            if performance_metrics:
                stage_info['performance_metrics'] = performance_metrics
            
            # Calculate stage efficiency
            efficiency = self._calculate_stage_efficiency(stage_info)
            stage_info['efficiency_score'] = efficiency
            
            # Calculate duration
            duration = (stage_info['end_time'] - stage_info['start_time']).total_seconds()
            
            summary = {
                'stage': stage_name,
                'duration_seconds': duration,
                'efficiency_score': efficiency,
                'resource_utilization': self._calculate_resource_utilization(stage_info),
                'performance_metrics': performance_metrics or {},
                'recommendations': self._generate_stage_recommendations(stage_info)
            }
            
            # Store in stage performance history
            self.stage_performance[stage_name] = summary
            
            logger.info(f"⏹️ Completed monitoring stage: {stage_name} (Efficiency: {efficiency:.2f})")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Stage monitoring end failed for {stage_name}: {e}")
            return {}
    
    def _collect_current_resource_state(self) -> Dict[str, Any]:
        """
        📸 รวบรวมสถานะทรัพยากรปัจจุบัน
        """
        try:
            import psutil
            
            state = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            }
            
            # Add GPU state if available
            if self.system_info.get('gpu', {}).get('cuda_available', False):
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_stats(0)
                        state['gpu_memory_used_gb'] = round(gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3), 2)
                        state['gpu_utilization'] = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                except:
                    pass
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Resource state collection failed: {e}")
            return {}
    
    def _calculate_stage_efficiency(self, stage_info: Dict[str, Any]) -> float:
        """
        🎯 คำนวณประสิทธิภาพของขั้นตอน
        """
        try:
            resource_usage = stage_info.get('resource_usage', [])
            if len(resource_usage) < 2:
                return 0.0
            
            # Calculate average resource utilization
            cpu_values = [r.get('cpu_percent', 0) for r in resource_usage if 'cpu_percent' in r]
            memory_values = [r.get('memory_percent', 0) for r in resource_usage if 'memory_percent' in r]
            
            if not cpu_values or not memory_values:
                return 0.0
            
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_memory = sum(memory_values) / len(memory_values)
            
            # Calculate duration
            start_time = stage_info.get('start_time')
            end_time = stage_info.get('end_time')
            if not start_time or not end_time:
                return 0.0
            
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Efficiency formula: balanced resource utilization vs time
            # Higher resource utilization = good, but excessive duration = bad
            target_cpu = 70  # Target CPU utilization
            target_memory = 60  # Target memory utilization
            
            cpu_efficiency = 1.0 - abs(avg_cpu - target_cpu) / 100
            memory_efficiency = 1.0 - abs(avg_memory - target_memory) / 100
            
            # Time factor (assume baseline of 5 minutes for normalization)
            time_factor = min(1.0, 5.0 / max(duration_minutes, 0.1))
            
            efficiency = (cpu_efficiency + memory_efficiency + time_factor) / 3
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            logger.error(f"❌ Stage efficiency calculation failed: {e}")
            return 0.0
    
    def _calculate_resource_utilization(self, stage_info: Dict[str, Any]) -> Dict[str, float]:
        """
        📊 คำนวณการใช้ทรัพยากรของขั้นตอน
        """
        try:
            resource_usage = stage_info.get('resource_usage', [])
            if not resource_usage:
                return {}
            
            # Extract all metrics
            cpu_values = [r.get('cpu_percent', 0) for r in resource_usage if 'cpu_percent' in r]
            memory_values = [r.get('memory_percent', 0) for r in resource_usage if 'memory_percent' in r]
            
            utilization = {}
            
            if cpu_values:
                utilization['cpu'] = {
                    'avg': round(sum(cpu_values) / len(cpu_values), 2),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                }
            
            if memory_values:
                utilization['memory'] = {
                    'avg': round(sum(memory_values) / len(memory_values), 2),
                    'max': max(memory_values),
                    'min': min(memory_values)
                }
            
            return utilization
            
        except Exception as e:
            logger.error(f"❌ Resource utilization calculation failed: {e}")
            return {}
    
    def _generate_stage_recommendations(self, stage_info: Dict[str, Any]) -> List[str]:
        """
        💡 สร้างคำแนะนำสำหรับการปรับปรุงขั้นตอน
        """
        recommendations = []
        
        try:
            utilization = self._calculate_resource_utilization(stage_info)
            efficiency = stage_info.get('efficiency_score', 0)
            
            # CPU recommendations
            cpu_util = utilization.get('cpu', {})
            if cpu_util.get('avg', 0) > 90:
                recommendations.append("🔥 High CPU usage - consider reducing batch size or parallel workers")
            elif cpu_util.get('avg', 0) < 30:
                recommendations.append("⚡ Low CPU usage - consider increasing batch size or parallel workers")
            
            # Memory recommendations
            memory_util = utilization.get('memory', {})
            if memory_util.get('avg', 0) > 85:
                recommendations.append("🧠 High memory usage - consider reducing data chunk size")
            elif memory_util.get('avg', 0) < 40:
                recommendations.append("📈 Low memory usage - can increase data processing chunk size")
            
            # Efficiency recommendations
            if efficiency < 0.5:
                recommendations.append("🎯 Low efficiency detected - review resource allocation strategy")
            elif efficiency > 0.8:
                recommendations.append("✨ Excellent efficiency - current configuration is optimal")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Stage recommendations generation failed: {e}")
            return []
    
    def _start_advanced_monitoring(self) -> None:
        """
        🚀 เริ่มการติดตามขั้นสูง
        """
        try:
            if not self.monitoring_active:
                self.start_monitoring(interval=0.5)  # More frequent monitoring
            
            # Start performance alert monitoring
            self.alert_thread = threading.Thread(target=self._performance_alert_loop)
            self.alert_thread.daemon = True
            self.alert_thread.start()
            
            logger.info("🚀 Advanced monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Advanced monitoring start failed: {e}")
    
    def _performance_alert_loop(self) -> None:
        """
        🚨 Loop การตรวจสอบและแจ้งเตือนประสิทธิภาพ
        """
        while self.monitoring_active:
            try:
                current_perf = self.get_current_performance()
                
                # Check for performance issues
                cpu_percent = current_perf.get('cpu_percent', 0)
                memory_percent = current_perf.get('memory', {}).get('percent', 0)
                
                # CPU alerts
                if cpu_percent > 95:
                    self.performance_alerts.put({
                        'type': 'critical',
                        'message': f'🔥 Critical CPU usage: {cpu_percent:.1f}%',
                        'timestamp': datetime.now().isoformat(),
                        'recommendation': 'Reduce computational load immediately'
                    })
                elif cpu_percent > 85:
                    self.performance_alerts.put({
                        'type': 'warning',
                        'message': f'⚠️ High CPU usage: {cpu_percent:.1f}%',
                        'timestamp': datetime.now().isoformat(),
                        'recommendation': 'Consider reducing batch size'
                    })
                
                # Memory alerts
                if memory_percent > 90:
                    self.performance_alerts.put({
                        'type': 'critical',
                        'message': f'🧠 Critical memory usage: {memory_percent:.1f}%',
                        'timestamp': datetime.now().isoformat(),
                        'recommendation': 'Reduce data chunk size immediately'
                    })
                elif memory_percent > 80:
                    self.performance_alerts.put({
                        'type': 'warning',
                        'message': f'⚠️ High memory usage: {memory_percent:.1f}%',
                        'timestamp': datetime.now().isoformat(),
                        'recommendation': 'Monitor memory usage closely'
                    })
                
                time.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Performance alert loop error: {e}")
                time.sleep(5.0)
    
    def get_performance_alerts(self, max_alerts: int = 10) -> List[Dict[str, Any]]:
        """
        🚨 รับการแจ้งเตือนประสิทธิภาพ
        
        Args:
            max_alerts: จำนวนการแจ้งเตือนสูงสุดที่ต้องการ
            
        Returns:
            List: รายการการแจ้งเตือน
        """
        alerts = []
        try:
            while not self.performance_alerts.empty() and len(alerts) < max_alerts:
                alerts.append(self.performance_alerts.get_nowait())
        except queue.Empty:
            pass
        
        return alerts
    
    def adaptive_resource_adjustment(self, current_stage: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔄 ปรับแต่งทรัพยากรแบบอัตโนมัติตามประสิทธิภาพ
        
        Args:
            current_stage: ขั้นตอนปัจจุบัน
            performance_metrics: เมตริกส์ประสิทธิภาพ
            
        Returns:
            Dict: การปรับแต่งที่แนะนำ
        """
        try:
            current_config = self.get_menu1_optimization_config()
            adjustments = {}
            
            # Analyze current performance
            current_perf = self.get_current_performance()
            cpu_percent = current_perf.get('cpu_percent', 0)
            memory_percent = current_perf.get('memory', {}).get('percent', 0)
            
            # Stage-specific adjustments
            if current_stage in ['feature_selection', 'cnn_lstm_training', 'dqn_training']:
                # High computational stages
                if cpu_percent > 90:
                    # Reduce batch size
                    current_batch = current_config.get(current_stage, {}).get('batch_size', 32)
                    new_batch = max(8, int(current_batch * 0.8))
                    adjustments['batch_size'] = new_batch
                    
                elif cpu_percent < 50 and memory_percent < 70:
                    # Can increase batch size
                    current_batch = current_config.get(current_stage, {}).get('batch_size', 32)
                    new_batch = min(256, int(current_batch * 1.2))
                    adjustments['batch_size'] = new_batch
            
            elif current_stage in ['data_loading', 'feature_engineering']:
                # I/O intensive stages
                if memory_percent > 85:
                    # Reduce chunk size
                    current_chunk = current_config.get('data_processing', {}).get('chunk_size', 4000)
                    new_chunk = max(1000, int(current_chunk * 0.7))
                    adjustments['chunk_size'] = new_chunk
                    
                elif memory_percent < 60:
                    # Can increase chunk size
                    current_chunk = current_config.get('data_processing', {}).get('chunk_size', 4000)
                    new_chunk = min(10000, int(current_chunk * 1.3))
                    adjustments['chunk_size'] = new_chunk
            
            # Record adjustment
            if adjustments:
                adjustment_record = {
                    'timestamp': datetime.now().isoformat(),
                    'stage': current_stage,
                    'adjustments': adjustments,
                    'trigger_metrics': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent
                    },
                    'performance_metrics': performance_metrics
                }
                self.adaptive_adjustments.append(adjustment_record)
                
                logger.info(f"🔄 Adaptive adjustment for {current_stage}: {adjustments}")
            
            return adjustments
            
        except Exception as e:
            logger.error(f"❌ Adaptive resource adjustment failed: {e}")
            return {}
    
    def generate_pipeline_performance_report(self) -> Dict[str, Any]:
        """
        📋 สร้างรายงานประสิทธิภาพ pipeline แบบครอบคลุม
        """
        try:
            # Base report from parent class
            base_report = self.generate_performance_report()
            
            # Enhanced pipeline-specific information
            pipeline_report = {
                **base_report,
                'pipeline_stages': self.stage_performance,
                'pipeline_summary': {
                    'total_stages': len(self.menu1_stages),
                    'completed_stages': len([s for s in self.stage_performance.values() if s.get('efficiency_score', 0) > 0]),
                    'average_efficiency': self._calculate_average_pipeline_efficiency(),
                    'total_duration_minutes': self._calculate_total_pipeline_duration(),
                    'resource_efficiency_grade': self._calculate_resource_efficiency_grade()
                },
                'performance_alerts_summary': {
                    'total_alerts': len(self.adaptive_adjustments),
                    'critical_alerts': len([a for a in self.adaptive_adjustments if 'critical' in str(a)]),
                    'recent_adjustments': self.adaptive_adjustments[-5:] if self.adaptive_adjustments else []
                },
                'optimization_recommendations': self._generate_pipeline_optimization_recommendations(),
                'menu1_integration': {
                    'active': self.menu1_integration_active,
                    'configuration_applied': bool(hasattr(self, 'menu1_instance')),
                    'monitoring_stages': list(self.pipeline_stage_monitoring.keys())
                }
            }
            
            return pipeline_report
            
        except Exception as e:
            logger.error(f"❌ Pipeline performance report generation failed: {e}")
            return {}
    
    def _calculate_average_pipeline_efficiency(self) -> float:
        """
        📊 คำนวณประสิทธิภาพเฉลี่ยของ pipeline
        """
        try:
            efficiencies = [stage.get('efficiency_score', 0) for stage in self.stage_performance.values()]
            return round(sum(efficiencies) / len(efficiencies), 3) if efficiencies else 0.0
        except:
            return 0.0
    
    def _calculate_total_pipeline_duration(self) -> float:
        """
        ⏱️ คำนวณระยะเวลารวมของ pipeline
        """
        try:
            durations = [stage.get('duration_seconds', 0) for stage in self.stage_performance.values()]
            return round(sum(durations) / 60, 2)  # Convert to minutes
        except:
            return 0.0
    
    def _calculate_resource_efficiency_grade(self) -> str:
        """
        🏆 คำนวณเกรดประสิทธิภาพทรัพยากร
        """
        try:
            avg_efficiency = self._calculate_average_pipeline_efficiency()
            
            if avg_efficiency >= 0.85:
                return "A+ (Excellent)"
            elif avg_efficiency >= 0.75:
                return "A (Very Good)"
            elif avg_efficiency >= 0.65:
                return "B (Good)"
            elif avg_efficiency >= 0.55:
                return "C (Fair)"
            else:
                return "D (Needs Improvement)"
                
        except:
            return "Unknown"
    
    def _generate_pipeline_optimization_recommendations(self) -> List[str]:
        """
        💡 สร้างคำแนะนำการปรับปรุง pipeline
        """
        recommendations = []
        
        try:
            avg_efficiency = self._calculate_average_pipeline_efficiency()
            total_duration = self._calculate_total_pipeline_duration()
            
            # General pipeline recommendations
            if avg_efficiency < 0.6:
                recommendations.append("🎯 Overall pipeline efficiency is low - review resource allocation strategy")
            
            if total_duration > 30:  # More than 30 minutes
                recommendations.append("⏰ Pipeline duration is long - consider increasing computational resources")
            
            # Stage-specific recommendations
            for stage_name, stage_data in self.stage_performance.items():
                efficiency = stage_data.get('efficiency_score', 0)
                duration = stage_data.get('duration_seconds', 0)
                
                if efficiency < 0.5:
                    recommendations.append(f"🔧 Stage '{stage_name}' has low efficiency - needs optimization")
                
                if duration > 600:  # More than 10 minutes
                    recommendations.append(f"⏱️ Stage '{stage_name}' is slow - consider resource adjustment")
            
            # Resource-based recommendations
            if self.system_info.get('memory', {}).get('total_gb', 0) < 8:
                recommendations.append("🧠 Consider upgrading RAM for better pipeline performance")
            
            if not self.system_info.get('gpu', {}).get('cuda_available', False):
                recommendations.append("🚀 Consider adding GPU acceleration for CNN-LSTM and DQN training")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Pipeline optimization recommendations failed: {e}")
            return []
    
    def display_real_time_dashboard(self) -> None:
        """
        📊 แสดง dashboard แบบ real-time
        """
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*80)
            print("⚡ ENHANCED INTELLIGENT RESOURCE MANAGEMENT DASHBOARD")
            print("="*80)
            
            # Current performance
            current_perf = self.get_current_performance()
            print(f"\n📊 REAL-TIME PERFORMANCE:")
            print(f"   CPU: {current_perf.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {current_perf.get('memory', {}).get('percent', 0):.1f}%")
            print(f"   Uptime: {current_perf.get('uptime_minutes', 0):.1f} minutes")
            
            # Pipeline progress
            if self.stage_performance:
                print(f"\n🌊 PIPELINE PROGRESS:")
                for stage, data in self.stage_performance.items():
                    efficiency = data.get('efficiency_score', 0)
                    duration = data.get('duration_seconds', 0)
                    status = "✅ Completed" if efficiency > 0 else "⏳ Pending"
                    print(f"   {stage}: {status} (Efficiency: {efficiency:.2f}, Duration: {duration:.1f}s)")
            
            # Recent alerts
            alerts = self.get_performance_alerts(5)
            if alerts:
                print(f"\n🚨 RECENT ALERTS:")
                for alert in alerts[-3:]:  # Show last 3 alerts
                    print(f"   {alert.get('type', 'INFO').upper()}: {alert.get('message', 'N/A')}")
            
            # Resource allocation
            optimization = self.resource_config.get('optimization', {})
            print(f"\n⚡ CURRENT ALLOCATION:")
            print(f"   Batch Size: {optimization.get('batch_size', 'N/A')}")
            print(f"   Workers: {optimization.get('recommended_workers', 'N/A')}")
            print(f"   Memory Limit: {optimization.get('memory_limit_gb', 'N/A'):.1f} GB")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Dashboard display failed: {e}")


def initialize_enhanced_intelligent_resources(allocation_percentage: float = 0.8,
                                              enable_advanced_monitoring: bool = True) -> EnhancedIntelligentResourceManager:
    """
    🚀 Initialize Enhanced Intelligent Resource Management System
    
    Args:
        allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
        enable_advanced_monitoring: เปิดใช้งานการติดตามขั้นสูง
        
    Returns:
        EnhancedIntelligentResourceManager: ตัวจัดการทรัพยากรขั้นสูง
    """
    try:
        # Initialize enhanced resource manager
        resource_manager = EnhancedIntelligentResourceManager(
            allocation_percentage=allocation_percentage,
            enable_advanced_monitoring=enable_advanced_monitoring
        )
        
        # Apply optimization
        resource_manager.apply_resource_optimization()
        
        # Display system summary
        resource_manager.display_system_summary()
        
        logger.info("🚀 Enhanced Intelligent Resource Management System ready")
        return resource_manager
        
    except Exception as e:
        logger.error(f"❌ Enhanced resource management initialization failed: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    try:
        print("⚡ Testing Enhanced Intelligent Resource Management System...")
        
        # Initialize enhanced system
        enhanced_manager = initialize_enhanced_intelligent_resources()
        
        # Simulate Menu 1 integration
        print("\n🌊 Simulating Menu 1 pipeline stages...")
        
        stages = ['data_loading', 'feature_engineering', 'cnn_lstm_training']
        for stage in stages:
            print(f"\n▶️ Starting {stage}...")
            enhanced_manager.start_stage_monitoring(stage)
            
            # Simulate some work
            time.sleep(2)
            
            # Simulate performance metrics
            metrics = {
                'auc': 0.75 + np.random.random() * 0.2,
                'accuracy': 0.80 + np.random.random() * 0.15,
                'duration': np.random.random() * 100 + 50
            }
            
            summary = enhanced_manager.end_stage_monitoring(stage, metrics)
            print(f"⏹️ {stage} completed - Efficiency: {summary.get('efficiency_score', 0):.2f}")
        
        # Generate comprehensive report
        report = enhanced_manager.generate_pipeline_performance_report()
        print(f"\n📋 Pipeline Summary:")
        pipeline_summary = report.get('pipeline_summary', {})
        print(f"   Average Efficiency: {pipeline_summary.get('average_efficiency', 0):.2f}")
        print(f"   Resource Grade: {pipeline_summary.get('resource_efficiency_grade', 'N/A')}")
        
        # Stop monitoring
        enhanced_manager.stop_monitoring()
        
        print("\n✅ Enhanced Intelligent Resource Management System test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
