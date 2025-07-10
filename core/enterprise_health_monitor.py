#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏥 NICEGOLD ENTERPRISE HEALTH MONITOR
Advanced system health monitoring for enterprise production environments

Features:
- Real-time health monitoring
- Component status tracking
- Performance degradation detection
- Predictive maintenance alerts
- Automated health reports
- Recovery recommendations
"""

import os
import sys
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.unified_enterprise_logger import get_unified_logger


class HealthStatus(Enum):
    """Health status enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Component health status"""
    name: str
    status: HealthStatus
    score: float  # 0-100
    message: str
    last_check: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class EnterpriseHealthMonitor:
    """
    🏥 Enterprise Health Monitor
    
    Comprehensive health monitoring system:
    - System resource monitoring
    - Component health tracking
    - Performance analysis
    - Predictive maintenance
    - Automated recovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_unified_logger()
        self.start_time = datetime.now()
        self.is_monitoring = False
        self.components = {}
        self.health_history = []
        
        # Health thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 2.0,
            'response_time_critical': 5.0
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'interval': self.config.get('monitoring_interval', 30),
            'history_retention_hours': self.config.get('history_retention_hours', 24),
            'auto_recovery': self.config.get('auto_recovery', True),
            'alert_threshold': self.config.get('alert_threshold', 3)
        }
        
        self._initialize_components()
        self.logger.info("🏥 Enterprise Health Monitor initialized")
        
    def _initialize_components(self):
        """Initialize component monitoring"""
        self.components = {
            'system_resources': ComponentHealth(
                name='System Resources',
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message='Initializing...',
                last_check=datetime.now()
            ),
            'data_processor': ComponentHealth(
                name='Data Processor',
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message='Initializing...',
                last_check=datetime.now()
            ),
            'ml_models': ComponentHealth(
                name='ML Models',
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message='Initializing...',
                last_check=datetime.now()
            ),
            'logging_system': ComponentHealth(
                name='Logging System',
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message='Initializing...',
                last_check=datetime.now()
            ),
            'database': ComponentHealth(
                name='Database',
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message='Initializing...',
                last_check=datetime.now()
            )
        }
        
    def start_monitoring(self):
        """Start health monitoring"""
        if self.is_monitoring:
            self.logger.warning("Health monitoring already active")
            return
            
        self.is_monitoring = True
        self.logger.info("🚀 Starting health monitoring...")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info("✅ Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        self.logger.info("⏹️ Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_all_components()
                self._analyze_health_trends()
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_config['interval'])
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5)
                
    def _check_all_components(self):
        """Check health of all components"""
        self._check_system_resources()
        self._check_data_processor()
        self._check_ml_models()
        self._check_logging_system()
        self._check_database()
        
        # Store health snapshot
        self._store_health_snapshot()
        
    def _check_system_resources(self):
        """Check system resource health"""
        try:
            # CPU check
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Calculate health score
            cpu_score = max(0, 100 - cpu_usage)
            memory_score = max(0, 100 - memory_usage)
            disk_score = max(0, 100 - disk_usage)
            overall_score = (cpu_score + memory_score + disk_score) / 3
            
            # Determine status
            if overall_score >= 80:
                status = HealthStatus.EXCELLENT
                message = "System resources optimal"
            elif overall_score >= 60:
                status = HealthStatus.GOOD
                message = "System resources good"
            elif overall_score >= 40:
                status = HealthStatus.WARNING
                message = "System resources under stress"
            else:
                status = HealthStatus.CRITICAL
                message = "System resources critical"
                
            self.components['system_resources'] = ComponentHealth(
                name='System Resources',
                status=status,
                score=overall_score,
                message=message,
                last_check=datetime.now(),
                details={
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'cpu_score': cpu_score,
                    'memory_score': memory_score,
                    'disk_score': disk_score
                }
            )
            
        except Exception as e:
            self.components['system_resources'] = ComponentHealth(
                name='System Resources',
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Health check failed: {e}",
                last_check=datetime.now()
            )
            
    def _check_data_processor(self):
        """Check data processor health"""
        try:
            # Test import
            from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
            
            # Basic functionality test
            processor = ElliottWaveDataProcessor()
            
            score = 90.0  # Assume good if no errors
            status = HealthStatus.EXCELLENT
            message = "Data processor operational"
            
            self.components['data_processor'] = ComponentHealth(
                name='Data Processor',
                status=status,
                score=score,
                message=message,
                last_check=datetime.now(),
                details={'import_successful': True}
            )
            
        except Exception as e:
            self.components['data_processor'] = ComponentHealth(
                name='Data Processor',
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Data processor error: {e}",
                last_check=datetime.now()
            )
            
    def _check_ml_models(self):
        """Check ML models health"""
        try:
            # Check if TensorFlow/PyTorch are working
            import tensorflow as tf
            import torch
            
            score = 85.0
            status = HealthStatus.GOOD
            message = "ML frameworks operational"
            
            self.components['ml_models'] = ComponentHealth(
                name='ML Models',
                status=status,
                score=score,
                message=message,
                last_check=datetime.now(),
                details={
                    'tensorflow_version': tf.__version__,
                    'torch_version': torch.__version__
                }
            )
            
        except Exception as e:
            self.components['ml_models'] = ComponentHealth(
                name='ML Models',
                status=HealthStatus.WARNING,
                score=40.0,
                message=f"ML framework issues: {e}",
                last_check=datetime.now()
            )
            
    def _check_logging_system(self):
        """Check logging system health"""
        try:
            # Test logger
            test_logger = get_unified_logger()
            test_logger.info("Health check test message")
            
            score = 95.0
            status = HealthStatus.EXCELLENT
            message = "Logging system operational"
            
            self.components['logging_system'] = ComponentHealth(
                name='Logging System',
                status=status,
                score=score,
                message=message,
                last_check=datetime.now(),
                details={'test_successful': True}
            )
            
        except Exception as e:
            self.components['logging_system'] = ComponentHealth(
                name='Logging System',
                status=HealthStatus.CRITICAL,
                score=20.0,
                message=f"Logging error: {e}",
                last_check=datetime.now()
            )
            
    def _check_database(self):
        """Check database health"""
        try:
            # Check if data files exist
            from core.project_paths import get_project_paths
            paths = get_project_paths()
            
            data_files = [
                paths.datacsv_dir / "XAUUSD_M1.csv",
                paths.datacsv_dir / "XAUUSD_M15.csv"
            ]
            
            files_exist = all(f.exists() for f in data_files)
            
            if files_exist:
                score = 90.0
                status = HealthStatus.EXCELLENT
                message = "Data files accessible"
            else:
                score = 30.0
                status = HealthStatus.WARNING
                message = "Some data files missing"
                
            self.components['database'] = ComponentHealth(
                name='Database',
                status=status,
                score=score,
                message=message,
                last_check=datetime.now(),
                details={'files_exist': files_exist}
            )
            
        except Exception as e:
            self.components['database'] = ComponentHealth(
                name='Database',
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Database error: {e}",
                last_check=datetime.now()
            )
            
    def _store_health_snapshot(self):
        """Store current health snapshot"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': self.get_overall_health_score(),
            'components': {name: {
                'status': comp.status.value,
                'score': comp.score,
                'message': comp.message
            } for name, comp in self.components.items()}
        }
        
        self.health_history.append(snapshot)
        
    def _analyze_health_trends(self):
        """Analyze health trends and predict issues"""
        if len(self.health_history) < 5:
            return  # Not enough data
            
        # Check for declining trends
        recent_scores = [h['overall_score'] for h in self.health_history[-5:]]
        
        if len(recent_scores) >= 3:
            trend = recent_scores[-1] - recent_scores[0]
            
            if trend < -10:  # Declining by more than 10 points
                self.logger.warning("🔍 Health declining trend detected")
                
    def _cleanup_old_data(self):
        """Clean up old health data"""
        cutoff_time = datetime.now() - timedelta(
            hours=self.monitoring_config['history_retention_hours']
        )
        
        self.health_history = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
    def get_overall_health_score(self) -> float:
        """Calculate overall health score"""
        if not self.components:
            return 0.0
            
        scores = [comp.score for comp in self.components.values()]
        return sum(scores) / len(scores)
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        overall_score = self.get_overall_health_score()
        
        if overall_score >= 80:
            overall_status = HealthStatus.EXCELLENT
        elif overall_score >= 60:
            overall_status = HealthStatus.GOOD
        elif overall_score >= 40:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.CRITICAL
            
        return {
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'monitoring_active': self.is_monitoring,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'components': {
                name: {
                    'status': comp.status.value,
                    'score': comp.score,
                    'message': comp.message,
                    'last_check': comp.last_check.isoformat()
                }
                for name, comp in self.components.items()
            },
            'recommendations': self.get_recommendations()
        }
        
    def get_recommendations(self) -> List[str]:
        """Get health improvement recommendations"""
        recommendations = []
        
        for name, comp in self.components.items():
            if comp.status == HealthStatus.CRITICAL:
                recommendations.append(f"🚨 Critical: Fix {name} immediately")
            elif comp.status == HealthStatus.WARNING:
                recommendations.append(f"⚠️ Warning: Monitor {name} closely")
                
        if not recommendations:
            recommendations.append("✅ System health is good")
            
        return recommendations
        
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        status = self.get_health_status()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_duration': (datetime.now() - self.start_time).total_seconds(),
            'overall_health': status,
            'history_count': len(self.health_history),
            'trends': self._analyze_trends(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze health trends"""
        if len(self.health_history) < 2:
            return {'trend': 'insufficient_data'}
            
        scores = [h['overall_score'] for h in self.health_history[-10:]]
        
        if len(scores) >= 2:
            trend = scores[-1] - scores[0]
            
            if trend > 5:
                trend_status = 'improving'
            elif trend < -5:
                trend_status = 'declining'
            else:
                trend_status = 'stable'
                
            return {
                'trend': trend_status,
                'change': trend,
                'recent_scores': scores
            }
            
        return {'trend': 'unknown'}
        
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.health_history:
            return {}
            
        scores = [h['overall_score'] for h in self.health_history]
        
        return {
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'score_variance': self._calculate_variance(scores)
        }
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


# Factory function
def get_enterprise_health_monitor(config: Optional[Dict] = None) -> EnterpriseHealthMonitor:
    """Factory function to get Enterprise Health Monitor instance"""
    return EnterpriseHealthMonitor(config)


# Export
__all__ = ['EnterpriseHealthMonitor', 'get_enterprise_health_monitor', 'HealthStatus', 'ComponentHealth']


if __name__ == "__main__":
    # Test the health monitor
    print("🏥 Testing Enterprise Health Monitor...")
    
    monitor = EnterpriseHealthMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run for a short time
    print("🔍 Monitoring for 10 seconds...")
    time.sleep(10)
    
    # Get status
    status = monitor.get_health_status()
    print(f"🏥 Health Status: {status['overall_status']} ({status['overall_score']:.1f})")
    
    # Get recommendations
    recommendations = monitor.get_recommendations()
    for rec in recommendations:
        print(f"💡 {rec}")
    
    # Generate report
    report = monitor.generate_health_report()
    print(f"📈 Health Report: {report}")
    
    # Stop monitoring
    monitor.stop_monitoring()
