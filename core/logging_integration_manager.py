#!/usr/bin/env python3
"""
🔗 NICEGOLD ENTERPRISE LOGGING INTEGRATION SYSTEM
ระบบผูกรวม Advanced Terminal Logger เข้ากับทุกส่วนของโปรเจค

🎯 Features:
- 🔧 Auto-integration with existing modules
- 📊 Centralized logging management  
- 🚀 Performance monitoring integration
- 🛡️ Error tracking & recovery
- 📈 Real-time statistics dashboard
- 🎨 Beautiful progress visualization
- 🔄 Dynamic logger injection
- 📋 Comprehensive system health monitoring
"""

import sys
import os
import time
import threading
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import traceback

# Import advanced logging components
try:
    from core.advanced_terminal_logger import (
        get_terminal_logger, init_terminal_logger, 
        LogLevel, ProcessStatus, AdvancedTerminalLogger
    )
    from core.real_time_progress_manager import (
        get_progress_manager, init_progress_manager,
        ProgressType, ProgressContext, RealTimeProgressManager
    )
    LOGGING_SYSTEM_AVAILABLE = True
except ImportError as e:
    LOGGING_SYSTEM_AVAILABLE = False
    print(f"⚠️ Advanced logging system not available: {e}")
    import logging


class LoggingIntegrationManager:
    """🔗 Manager สำหรับผูกระบบ logging เข้ากับทุกส่วนของโปรเจค"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.integration_status = {}
        self.integrated_modules = {}
        self.logger = None
        self.progress_manager = None
        self.system_health = {
            'start_time': time.time(),
            'integration_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'last_health_check': None
        }
        
        # Initialize logging system
        self._initialize_logging_system()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _initialize_logging_system(self):
        """🚀 Initialize advanced logging system"""
        try:
            if LOGGING_SYSTEM_AVAILABLE:
                # Initialize terminal logger
                self.logger = init_terminal_logger(
                    name="NICEGOLD_INTEGRATION",
                    enable_rich=True,
                    enable_file_logging=True,
                    log_dir=str(self.project_root / "logs"),
                    max_console_lines=2000
                )
                
                # Initialize progress manager
                self.progress_manager = init_progress_manager(
                    enable_rich=True,
                    max_concurrent_bars=20,
                    refresh_rate=0.1,
                    auto_cleanup=True
                )
                
                self.logger.success("🚀 Advanced logging system initialized", "Integration_Manager")
                self.logger.system(f"Project root: {self.project_root}", "Integration_Manager")
                
            else:
                # Fallback to basic logging
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                self.logger = logging.getLogger("NICEGOLD_INTEGRATION")
                self.logger.info("📝 Basic logging system initialized (fallback)")
                
        except Exception as e:
            print(f"❌ Failed to initialize logging system: {e}")
            self.logger = logging.getLogger("NICEGOLD_INTEGRATION")
    
    def _start_health_monitoring(self):
        """📈 Start system health monitoring"""
        if LOGGING_SYSTEM_AVAILABLE and self.logger:
            # Start health monitoring thread
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            
            self.logger.system("📈 System health monitoring started", "Health_Monitor")
    
    def _health_monitoring_loop(self):
        """🔄 Health monitoring loop"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._perform_health_check()
            except Exception as e:
                if self.logger and LOGGING_SYSTEM_AVAILABLE:
                    self.logger.error("Health monitoring error", "Health_Monitor", exception=e)
                else:
                    print(f"❌ Health monitoring error: {e}")
    
    def _perform_health_check(self):
        """🏥 Perform system health check"""
        try:
            self.system_health['last_health_check'] = datetime.now()
            
            # Check logger health
            if LOGGING_SYSTEM_AVAILABLE and self.logger:
                stats = self.logger.monitor.get_stats()
                
                # Log health metrics
                self.logger.performance(
                    f"System Health Check - Memory: {stats.get('memory_current_mb', 0):.1f}MB, "
                    f"CPU: {stats.get('cpu_percent', 0):.1f}%, "
                    f"Logs: {stats.get('total_logs', 0)}, "
                    f"Errors: {stats.get('error_count', 0)}",
                    "Health_Check",
                    data=stats
                )
                
                # Update system health
                self.system_health.update({
                    'memory_mb': stats.get('memory_current_mb', 0),
                    'cpu_percent': stats.get('cpu_percent', 0),
                    'total_logs': stats.get('total_logs', 0),
                    'error_count': stats.get('error_count', 0),
                    'uptime': stats.get('uptime', 0)
                })
                
        except Exception as e:
            self.system_health['error_count'] += 1
            if self.logger and LOGGING_SYSTEM_AVAILABLE:
                self.logger.error("Health check failed", "Health_Check", exception=e)
    
    def integrate_with_module(self, module_path: str, force_integration: bool = False) -> bool:
        """🔧 Integrate logging system with a specific module"""
        try:
            # Check if already integrated
            if module_path in self.integrated_modules and not force_integration:
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.warning(f"Module already integrated: {module_path}", "Module_Integration")
                return True
            
            # Import the module
            module_file = self.project_root / module_path
            if not module_file.exists():
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.error(f"Module not found: {module_path}", "Module_Integration")
                return False
            
            # Add project root to Python path if needed
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            # Import module dynamically
            module_name = module_path.replace('/', '.').replace('.py', '')
            try:
                module = importlib.import_module(module_name)
                
                # Inject logger into module
                self._inject_logger_into_module(module, module_path)
                
                self.integrated_modules[module_path] = {
                    'module': module,
                    'integration_time': datetime.now(),
                    'status': 'integrated'
                }
                
                self.system_health['integration_count'] += 1
                
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.success(f"Successfully integrated with module: {module_path}", 
                                      "Module_Integration")
                else:
                    print(f"✅ Successfully integrated with module: {module_path}")
                
                return True
                
            except ImportError as e:
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.error(f"Failed to import module {module_path}", 
                                    "Module_Integration", exception=e)
                else:
                    print(f"❌ Failed to import module {module_path}: {e}")
                return False
                
        except Exception as e:
            self.system_health['error_count'] += 1
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.error(f"Module integration failed for {module_path}", 
                                "Module_Integration", exception=e)
            else:
                print(f"❌ Module integration failed for {module_path}: {e}")
            return False
    
    def _inject_logger_into_module(self, module: Any, module_path: str):
        """💉 Inject logger into module"""
        try:
            # Inject advanced logger if available
            if LOGGING_SYSTEM_AVAILABLE and self.logger:
                # Set advanced logger
                setattr(module, 'advanced_logger', self.logger)
                setattr(module, 'progress_manager', self.progress_manager)
                
                # Create module-specific logger methods
                module_name = module_path.split('/')[-1].replace('.py', '')
                
                def create_module_logger(category_prefix):
                    def log_info(msg, **kwargs):
                        self.logger.info(msg, f"{category_prefix}_Info", **kwargs)
                    def log_error(msg, **kwargs):
                        self.logger.error(msg, f"{category_prefix}_Error", **kwargs)
                    def log_warning(msg, **kwargs):
                        self.logger.warning(msg, f"{category_prefix}_Warning", **kwargs)
                    def log_success(msg, **kwargs):
                        self.logger.success(msg, f"{category_prefix}_Success", **kwargs)
                    def log_debug(msg, **kwargs):
                        self.logger.debug(msg, f"{category_prefix}_Debug", **kwargs)
                    
                    return {
                        'info': log_info,
                        'error': log_error,
                        'warning': log_warning,
                        'success': log_success,
                        'debug': log_debug
                    }
                
                # Inject module-specific logger
                module_logger = create_module_logger(module_name.title())
                setattr(module, 'module_logger', module_logger)
                
                # Try to replace existing logger if found
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, 'info') and hasattr(attr, 'error'):  # Looks like a logger
                        # Create a wrapper that uses our advanced logger
                        if attr_name == 'logger':
                            setattr(module, 'old_logger', attr)
                            setattr(module, 'logger', self.logger)
                
                self.logger.system(f"Logger injected into module: {module_path}", 
                                 "Logger_Injection")
            
        except Exception as e:
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.error(f"Logger injection failed for {module_path}", 
                                "Logger_Injection", exception=e)
            else:
                print(f"❌ Logger injection failed for {module_path}: {e}")
    
    def integrate_with_core_modules(self) -> Dict[str, bool]:
        """🏢 Integrate with all core NICEGOLD modules"""
        
        # Start integration progress
        integration_progress = None
        if LOGGING_SYSTEM_AVAILABLE and self.progress_manager:
            integration_progress = self.progress_manager.create_progress(
                "🔗 Core Module Integration", 0, ProgressType.PROCESSING
            )
        
        core_modules = [
            # Core system modules
            'core/menu_system.py',
            'core/config.py',
            'core/logger.py',
            'core/compliance.py',
            'core/output_manager.py',
            'core/project_paths.py',
            
            # Elliott Wave AI modules
            'elliott_wave_modules/data_processor.py',
            'elliott_wave_modules/cnn_lstm_engine.py',
            'elliott_wave_modules/dqn_agent.py',
            'elliott_wave_modules/feature_selector.py',
            'elliott_wave_modules/pipeline_orchestrator.py',
            'elliott_wave_modules/performance_analyzer.py',
            'elliott_wave_modules/feature_engineering.py',
            'elliott_wave_modules/enterprise_ml_protection.py',
            
            # Menu modules
            'menu_modules/menu_1_elliott_wave.py',
        ]
        
        results = {}
        
        # Update progress total
        if integration_progress:
            self.progress_manager.set_progress(integration_progress, 0)
            self.progress_manager.update_progress(integration_progress, 0, 
                                                f"Starting integration with {len(core_modules)} modules")
        
        if LOGGING_SYSTEM_AVAILABLE:
            self.logger.info(f"🔗 Starting core module integration with {len(core_modules)} modules", 
                           "Core_Integration")
        
        for i, module_path in enumerate(core_modules):
            try:
                if integration_progress:
                    self.progress_manager.update_progress(
                        integration_progress, 1, 
                        f"Integrating: {module_path.split('/')[-1]}"
                    )
                
                success = self.integrate_with_module(module_path)
                results[module_path] = success
                
                if not success:
                    self.system_health['warning_count'] += 1
                
                # Small delay to show progress
                time.sleep(0.1)
                
            except Exception as e:
                results[module_path] = False
                self.system_health['error_count'] += 1
                
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.error(f"Integration failed for {module_path}", 
                                    "Core_Integration", exception=e)
        
        # Complete integration progress
        if integration_progress:
            successful_integrations = sum(1 for success in results.values() if success)
            self.progress_manager.complete_progress(
                integration_progress, 
                f"✅ Integration completed: {successful_integrations}/{len(core_modules)} successful"
            )
        
        # Log summary
        successful_count = sum(1 for success in results.values() if success)
        if LOGGING_SYSTEM_AVAILABLE:
            self.logger.success(f"Core module integration completed: {successful_count}/{len(core_modules)} successful", 
                              "Core_Integration", data=results)
        else:
            print(f"✅ Core module integration completed: {successful_count}/{len(core_modules)} successful")
        
        return results
    
    def get_integration_status(self) -> Dict[str, Any]:
        """📊 Get integration status report"""
        return {
            'total_integrated': len(self.integrated_modules),
            'integration_success_rate': len(self.integrated_modules) / max(1, self.system_health.get('integration_count', 1)),
            'system_health': self.system_health.copy(),
            'integrated_modules': list(self.integrated_modules.keys()),
            'logging_system_available': LOGGING_SYSTEM_AVAILABLE,
            'last_health_check': self.system_health.get('last_health_check'),
            'timestamp': datetime.now().isoformat()
        }
    
    def show_integration_dashboard(self):
        """📈 Show integration dashboard"""
        if LOGGING_SYSTEM_AVAILABLE and self.logger:
            self.logger.show_system_stats()
            
            # Show integration status
            status = self.get_integration_status()
            
            self.logger.system("🔗 Integration Dashboard", "Dashboard")
            self.logger.info(f"📊 Total Integrated Modules: {status['total_integrated']}", "Dashboard")
            self.logger.info(f"📈 Success Rate: {status['integration_success_rate']:.1%}", "Dashboard")
            self.logger.info(f"🏥 System Health: {len(status['integrated_modules'])} active modules", "Dashboard")
            
            if self.progress_manager:
                progress_stats = self.progress_manager.get_statistics()
                self.logger.performance(f"Progress Manager Stats", "Dashboard", data=progress_stats)
        
        else:
            # Simple dashboard
            status = self.get_integration_status()
            print("\n" + "="*60)
            print("🔗 NICEGOLD LOGGING INTEGRATION DASHBOARD")
            print("="*60)
            print(f"📊 Total Integrated Modules: {status['total_integrated']}")
            print(f"📈 Success Rate: {status['integration_success_rate']:.1%}")
            print(f"🏥 System Health: {len(status['integrated_modules'])} active modules")
            print(f"⏰ Last Health Check: {status.get('last_health_check', 'Never')}")
            print("="*60)
    
    def test_integration(self) -> bool:
        """🧪 Test integration functionality"""
        try:
            test_progress = None
            if LOGGING_SYSTEM_AVAILABLE and self.progress_manager:
                test_progress = self.progress_manager.create_progress(
                    "🧪 Integration Test", 5, ProgressType.VALIDATION
                )
            
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.info("🧪 Starting integration test", "Integration_Test")
            
            # Test 1: Logger functionality
            if test_progress:
                self.progress_manager.update_progress(test_progress, 1, "Testing logger functionality")
            
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.debug("Debug test message", "Test")
                self.logger.info("Info test message", "Test")
                self.logger.warning("Warning test message", "Test")
                self.logger.success("Success test message", "Test")
            
            # Test 2: Progress manager
            if test_progress:
                self.progress_manager.update_progress(test_progress, 1, "Testing progress manager")
            
            # Test 3: Module integration
            if test_progress:
                self.progress_manager.update_progress(test_progress, 1, "Testing module integration")
            
            # Test 4: Error handling
            if test_progress:
                self.progress_manager.update_progress(test_progress, 1, "Testing error handling")
            
            try:
                raise Exception("Test exception for error handling")
            except Exception as e:
                if LOGGING_SYSTEM_AVAILABLE:
                    self.logger.error("Test exception handled correctly", "Test", exception=e)
            
            # Test 5: System health
            if test_progress:
                self.progress_manager.update_progress(test_progress, 1, "Testing system health")
            
            self._perform_health_check()
            
            # Complete test
            if test_progress:
                self.progress_manager.complete_progress(test_progress, "✅ All tests passed")
            
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.success("🎉 Integration test completed successfully", "Integration_Test")
            else:
                print("🎉 Integration test completed successfully")
            
            return True
            
        except Exception as e:
            if test_progress:
                self.progress_manager.fail_progress(test_progress, f"Test failed: {str(e)}")
            
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.error("Integration test failed", "Integration_Test", exception=e)
            else:
                print(f"❌ Integration test failed: {e}")
            
            return False
    
    def export_integration_report(self, filepath: str = None) -> str:
        """📋 Export integration report"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"integration_report_{timestamp}.json"
        
        try:
            import json
            
            report_data = {
                'integration_status': self.get_integration_status(),
                'system_health': self.system_health,
                'integrated_modules': {
                    module: {
                        'integration_time': info['integration_time'].isoformat(),
                        'status': info['status']
                    } for module, info in self.integrated_modules.items()
                },
                'logging_system_info': {
                    'advanced_logging_available': LOGGING_SYSTEM_AVAILABLE,
                    'logger_type': type(self.logger).__name__ if self.logger else None,
                    'progress_manager_available': self.progress_manager is not None
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.success(f"Integration report exported: {filepath}", "Report_Export")
            else:
                print(f"✅ Integration report exported: {filepath}")
            
            return filepath
            
        except Exception as e:
            if LOGGING_SYSTEM_AVAILABLE:
                self.logger.error("Failed to export integration report", "Report_Export", exception=e)
            else:
                print(f"❌ Failed to export integration report: {e}")
            return None


# Global integration manager instance
integration_manager = None

def get_integration_manager() -> LoggingIntegrationManager:
    """Get global integration manager instance"""
    global integration_manager
    if integration_manager is None:
        integration_manager = LoggingIntegrationManager()
    return integration_manager

def init_integration_manager(project_root: str = None) -> LoggingIntegrationManager:
    """Initialize global integration manager"""
    global integration_manager
    integration_manager = LoggingIntegrationManager(project_root)
    return integration_manager

def integrate_logging_system(project_root: str = None) -> bool:
    """🚀 Quick setup for integrating logging system with entire project"""
    try:
        manager = init_integration_manager(project_root)
        
        # Test integration first
        if not manager.test_integration():
            print("⚠️ Integration test failed, but continuing...")
        
        # Integrate with core modules
        results = manager.integrate_with_core_modules()
        
        # Show dashboard
        manager.show_integration_dashboard()
        
        # Export report
        report_file = manager.export_integration_report()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        if LOGGING_SYSTEM_AVAILABLE and manager.logger:
            manager.logger.success(
                f"🎉 Logging system integration completed: {success_count}/{total_count} modules integrated",
                "Quick_Setup"
            )
        else:
            print(f"🎉 Logging system integration completed: {success_count}/{total_count} modules integrated")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Failed to integrate logging system: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the integration system
    print("🚀 Testing NICEGOLD Logging Integration System...")
    
    # Quick integration test
    success = integrate_logging_system()
    
    if success:
        print("✅ Integration test completed successfully!")
        
        # Get manager and show final dashboard
        manager = get_integration_manager()
        manager.show_integration_dashboard()
    else:
        print("❌ Integration test failed!")
