#!/usr/bin/env python3
"""
🔗 MENU 1 ENTERPRISE LOGGER INTEGRATION
ระบบผสานระบบ Logging ใหม่เข้ากับ Menu 1 Elliott Wave Pipeline

🎯 Integration Features:
- 🚀 Auto-detect Menu 1 implementations
- 🔄 Seamless logger replacement
- 📊 Enhanced progress tracking for all 10 steps
- 🛡️ Backward compatibility with existing loggers
- ⚡ Performance monitoring integration
- 🎨 Beautiful progress bars for each step
- 📁 Enterprise file management
- 🔍 Real-time error and warning tracking
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Type
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our enterprise logger
try:
    from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus
        get_menu1_logger, 
        Menu1Step, 
        Menu1LogLevel,
        Menu1ProcessStatus,
        EnterpriseMenu1TerminalLogger
    )
    ENTERPRISE_LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enterprise logger not available: {e}")
    ENTERPRISE_LOGGER_AVAILABLE = False


class Menu1LoggerIntegrator:
    """🔗 Menu 1 Logger Integration Manager"""
    
    def __init__(self):
        self.original_logger = None
        self.enterprise_logger = get_unified_logger()= None
        self.step_mapping = {
            1: Menu1Step.STEP_1,
            2: Menu1Step.STEP_2,
            3: Menu1Step.STEP_3,
            4: Menu1Step.STEP_4,
            5: Menu1Step.STEP_5,
            6: Menu1Step.STEP_6,
            7: Menu1Step.STEP_7,
            8: Menu1Step.STEP_8,
            9: Menu1Step.STEP_9,
            10: Menu1Step.STEP_10
        }
        
        # Initialize enterprise logger if available
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger = get_unified_logger()= menu1_instance.logger
            
            # Replace with our enterprise logger wrapper
            menu1_instance.logger = EnterpriseLoggerWrapper(
                self.enterprise_logger, 
                original_logger=self.original_logger
            )
            
            print("✅ Menu 1 integrated with Enterprise Terminal Logger")
            return True
            
        except Exception as e:
            print(f"❌ Menu 1 integration failed: {e}")
            return False
    
    def start_pipeline_step(self, step_number: int, total_operations: int = 100) -> Optional[str]:
        """เริ่มต้น pipeline step"""
        if not ENTERPRISE_LOGGER_AVAILABLE or step_number not in self.step_mapping:
            return None
        
        step = self.step_mapping[step_number]
        self.active_step_id = self.enterprise_logger.start_step(step, total_operations)
        return self.active_step_id
    
    def update_pipeline_progress(self, increment: int = 1, status: str = None, substep: str = None):
        """อัปเดต pipeline progress"""
        if self.active_step_id and ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger.update_step_progress(
                self.active_step_id, increment, status, substep
            )
    
    def complete_pipeline_step(self, success: bool = True, message: str = None, performance_data: Dict = None):
        """สิ้นสุด pipeline step"""
        if self.active_step_id and ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger.complete_step(
                self.active_step_id, success, message, performance_data
            )
            self.active_step_id = None
    
    def log_elliott_wave_progress(self, message: str, data: Dict = None):
        """Log Elliott Wave progress"""
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger.log_elliott_wave_progress(message, data)
    
    def log_ai_training_progress(self, message: str, metrics: Dict = None):
        """Log AI training progress"""
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger.log_ai_training_progress(message, metrics)
    
    def display_performance_dashboard(self):
        """แสดง performance dashboard"""
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.enterprise_logger.display_performance_dashboard()
    
    def generate_session_report(self) -> Optional[Dict]:
        """สร้าง session report"""
        if ENTERPRISE_LOGGER_AVAILABLE:
            return self.enterprise_logger.generate_session_report()
        return None


class EnterpriseLoggerWrapper:
    """🎭 Wrapper สำหรับ Enterprise Logger ที่ Compatible กับ Standard Logger"""
    
    def __init__(self, enterprise_logger: EnterpriseMenu1TerminalLogger, 
                 original_logger: Optional[logging.Logger] = None):
        self.enterprise_logger = get_unified_logger()= original_logger
        self.current_step = None
        self.step_progress = 0
    
    def info(self, message: str, context: str = "General"):
        """Log info message"""
        self.enterprise_logger.log(Menu1LogLevel.INFO, context, message)
        if self.original_logger:
            self.original_logger.info(f"{context}: {message}")
    
    def warning(self, message: str, context: str = "General"):
        """Log warning message"""
        self.enterprise_logger.log(Menu1LogLevel.WARNING, context, message)
        if self.original_logger:
            self.original_logger.warning(f"{context}: {message}")
    
    def error(self, message: str, context: str = "General", exception: Exception = None):
        """Log error message"""
        self.enterprise_logger.log(Menu1LogLevel.ERROR, context, message, exception=exception)
        if self.original_logger:
            if exception:
                self.original_logger.error(f"{context}: {message}", exc_info=exception)
            else:
                self.original_logger.error(f"{context}: {message}")
    
    def debug(self, message: str, context: str = "General"):
        """Log debug message"""
        self.enterprise_logger.log(Menu1LogLevel.DEBUG, context, message)
        if self.original_logger:
            self.original_logger.debug(f"{context}: {message}")
    
    def success(self, message: str, context: str = "General"):
        """Log success message"""
        self.enterprise_logger.log(Menu1LogLevel.SUCCESS, context, message)
        if self.original_logger:
            self.original_logger.info(f"SUCCESS {context}: {message}")
    
    def critical(self, message: str, context: str = "General", exception: Exception = None):
        """Log critical message"""
        self.enterprise_logger.log(Menu1LogLevel.CRITICAL, context, message, exception=exception)
        if self.original_logger:
            if exception:
                self.original_logger.critical(f"{context}: {message}", exc_info=exception)
            else:
                self.original_logger.critical(f"{context}: {message}")
    
    def log_step_start(self, step_num: int, step_name: str, description: str = ""):
        """เริ่มต้น step ใหม่"""
        if step_num in range(1, 11):
            integrator = Menu1LoggerIntegrator()
            self.current_step = integrator.start_pipeline_step(step_num, 100)
            self.step_progress = 0
            self.info(f"Started {step_name}", f"Step_{step_num}")
    
    def log_step_progress(self, message: str, progress: int = None):
        """อัปเดต step progress"""
        if self.current_step:
            if progress is not None:
                increment = progress - self.step_progress
                self.step_progress = progress
            else:
                increment = 1
                self.step_progress += 1
            
            integrator = Menu1LoggerIntegrator()
            integrator.active_step_id = self.current_step
            integrator.update_pipeline_progress(increment, message, message)
    
    def log_step_complete(self, step_num: int, success: bool = True, message: str = ""):
        """สิ้นสุด step"""
        if self.current_step:
            integrator = Menu1LoggerIntegrator()
            integrator.active_step_id = self.current_step
            integrator.complete_pipeline_step(success, message)
            self.current_step = None
            
            status = "completed successfully" if success else "failed"
            self.info(f"Step {step_num} {status}: {message}", f"Step_{step_num}")
    
    def log_elliott_wave(self, message: str, data: Dict = None):
        """Log Elliott Wave specific message"""
        self.enterprise_logger.log_elliott_wave_progress(message, data)
    
    def log_ai_training(self, message: str, metrics: Dict = None):
        """Log AI training specific message"""
        self.enterprise_logger.log_ai_training_progress(message, metrics)


def auto_integrate_menu1_modules():
    """อัตโนมัติผสาน enterprise logger กับ Menu 1 modules ทั้งหมด"""
    if not ENTERPRISE_LOGGER_AVAILABLE:
        print("⚠️ Enterprise logger not available for auto-integration")
        return False
    
    print("🔍 Scanning for Menu 1 implementations...")
    
    # List of possible Menu 1 module paths
    menu1_modules = [
        "menu_modules.enhanced_menu_1_elliott_wave",
        "menu_modules.menu_1_elliott_wave",
        "menu_modules.menu_1_elliott_wave_advanced",
        "menu_modules.enhanced_menu_1_elliott_wave_advanced",
        "menu_modules.enhanced_menu_1_elliott_wave_perfect",
        "menu_modules.completely_fixed_production_menu_1",
        "menu_modules.menu_1_elliott_wave_complete"
    ]
    
    integrated_count = 0
    
    for module_path in menu1_modules:
        try:
            module = importlib.import_module(module_path)
            
            # Look for Menu 1 classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    'menu' in attr_name.lower() and 
                    '1' in attr_name and
                    hasattr(attr, '__init__')):
                    
                    print(f"📦 Found Menu 1 class: {attr_name} in {module_path}")
                    
                    # Add integration method to class
                    original_init = attr.__init__
                    
                    def enhanced_init(self, *args, **kwargs):
                        original_init(self, *args, **kwargs)
                        integrator = Menu1LoggerIntegrator()
                        integrator.integrate_with_menu1_class(self)
                    
                    attr.__init__ = enhanced_init
                    integrated_count += 1
                    print(f"✅ Integrated {attr_name} with Enterprise Logger")
        
        except ImportError:
            continue
        except Exception as e:
            print(f"⚠️ Failed to integrate {module_path}: {e}")
    
    print(f"🎉 Auto-integration completed: {integrated_count} Menu 1 classes integrated")
    return integrated_count > 0


def create_enhanced_menu1_runner():
    """สร้าง enhanced Menu 1 runner ที่มี enterprise logging"""
    
    if not ENTERPRISE_LOGGER_AVAILABLE:
        print("❌ Enterprise logger not available")
        return None
    
    class EnhancedMenu1Runner:
        """🚀 Enhanced Menu 1 Runner with Enterprise Logging"""
        
        def __init__(self):
            self.enterprise_logger = get_unified_logger()= Menu1LoggerIntegrator()
        
        def run_menu1_with_enterprise_logging(self, menu1_class: Type, *args, **kwargs):
            """รัน Menu 1 พร้อม enterprise logging"""
            try:
                # Create Menu 1 instance
                menu1_instance = menu1_class(*args, **kwargs)
                
                # Integrate enterprise logger
                self.integrator.integrate_with_menu1_class(menu1_instance)
                
                # Run with enhanced logging
                self.enterprise_logger.log(
                    Menu1LogLevel.SYSTEM, 
                    "Menu1_Runner", 
                    f"Starting {menu1_class.__name__} with Enterprise Logging"
                )
                
                # Execute Menu 1
                if hasattr(menu1_instance, 'run'):
                    result = menu1_instance.run()
                elif hasattr(menu1_instance, 'run_enhanced_pipeline'):
                    result = menu1_instance.run_enhanced_pipeline()
                elif hasattr(menu1_instance, 'run_elliott_wave_pipeline'):
                    result = menu1_instance.run_elliott_wave_pipeline()
                else:
                    raise Exception("No suitable run method found in Menu 1 class")
                
                # Display final dashboard
                self.enterprise_logger.display_performance_dashboard()
                
                # Generate session report
                report = self.enterprise_logger.generate_session_report()
                
                self.enterprise_logger.log(
                    Menu1LogLevel.SUCCESS,
                    "Menu1_Runner", 
                    "Menu 1 execution completed successfully"
                )
                
                return {
                    'result': result,
                    'session_report': report,
                    'success': True
                }
                
            except Exception as e:
                self.enterprise_logger.log(
                    Menu1LogLevel.ERROR,
                    "Menu1_Runner",
                    f"Menu 1 execution failed: {str(e)}",
                    exception=e
                )
                
                return {
                    'result': None,
                    'error': str(e),
                    'success': False
                }
        
        def display_session_summary(self):
            """แสดงสรุป session"""
            self.enterprise_logger.display_session_summary()
    
    return EnhancedMenu1Runner()


def patch_existing_menu1_implementations():
    """Patch Menu 1 implementations ที่มีอยู่แล้วเพื่อใช้ enterprise logger"""
    
    if not ENTERPRISE_LOGGER_AVAILABLE:
        return False
    
    print("🔧 Patching existing Menu 1 implementations...")
    
    # Import and patch enhanced menu
    try:
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        
        # Add enterprise logging methods
        def log_step_start_enhanced(self, step_num: int, step_name: str, description: str = ""):
            if hasattr(self, 'logger') and hasattr(self.logger, 'log_step_start'):
                self.logger.log_step_start(step_num, step_name, description)
            else:
                print(f"🚀 Step {step_num}: {step_name}")
        
        def log_step_progress_enhanced(self, message: str, progress: int = None):
            if hasattr(self, 'logger') and hasattr(self.logger, 'log_step_progress'):
                self.logger.log_step_progress(message, progress)
            else:
                print(f"📊 Progress: {message}")
        
        def log_step_complete_enhanced(self, step_num: int, success: bool = True, message: str = ""):
            if hasattr(self, 'logger') and hasattr(self.logger, 'log_step_complete'):
                self.logger.log_step_complete(step_num, success, message)
            else:
                status = "✅" if success else "❌"
                print(f"{status} Step {step_num} completed: {message}")
        
        # Patch methods
        EnhancedMenu1ElliottWave.log_step_start = log_step_start_enhanced
        EnhancedMenu1ElliottWave.log_step_progress = log_step_progress_enhanced  
        EnhancedMenu1ElliottWave.log_step_complete = log_step_complete_enhanced
        
        print("✅ Patched EnhancedMenu1ElliottWave")
        
    except ImportError:
        print("⚠️ EnhancedMenu1ElliottWave not available for patching")
    
    # Patch other Menu 1 implementations
    menu_classes = [
        ("menu_modules.menu_1_elliott_wave", "Menu1ElliottWave"),
        ("menu_modules.menu_1_elliott_wave_advanced", "Menu1ElliottWaveAdvanced"),
        ("menu_modules.completely_fixed_production_menu_1", "CompletelyFixedProductionMenu1")
    ]
    
    patched_count = 0
    
    for module_path, class_name in menu_classes:
        try:
            module = importlib.import_module(module_path)
            menu_class = getattr(module, class_name)
            
            # Add enterprise logging methods
            def make_log_method(method_name):
                def log_method(self, *args, **kwargs):
                    if hasattr(self, 'logger') and hasattr(self.logger, method_name):
                        return getattr(self.logger, method_name)(*args, **kwargs)
                    else:
                        # Fallback logging
                        if method_name == 'log_step_start':
                            print(f"🚀 Step {args[0]}: {args[1]}")
                        elif method_name == 'log_step_progress':
                            print(f"📊 Progress: {args[0]}")
                        elif method_name == 'log_step_complete':
                            status = "✅" if args[1] else "❌"
                            print(f"{status} Step {args[0]} completed")
                return log_method
            
            menu_class.log_step_start = make_log_method('log_step_start')
            menu_class.log_step_progress = make_log_method('log_step_progress')
            menu_class.log_step_complete = make_log_method('log_step_complete')
            
            patched_count += 1
            print(f"✅ Patched {class_name}")
            
        except (ImportError, AttributeError):
            continue
        except Exception as e:
            print(f"⚠️ Failed to patch {class_name}: {e}")
    
    print(f"🎉 Patching completed: {patched_count} classes patched")
    return patched_count > 0


# Auto-integration on import
if __name__ != "__main__":
    # Auto-patch when imported
    try:
        patch_existing_menu1_implementations()
    except Exception as e:
        print(f"⚠️ Auto-patching failed: {e}")


# Export main classes and functions
__all__ = [
    'Menu1LoggerIntegrator',
    'EnterpriseLoggerWrapper', 
    'auto_integrate_menu1_modules',
    'create_enhanced_menu1_runner',
    'patch_existing_menu1_implementations'
]


if __name__ == "__main__":
    print("🔗 MENU 1 ENTERPRISE LOGGER INTEGRATION")
    print("=" * 60)
    
    # Test integration
    if ENTERPRISE_LOGGER_AVAILABLE:
        print("✅ Enterprise logger available")
        
        # Test auto-integration
        auto_integrate_menu1_modules()
        
        # Test enhanced runner
        runner = create_enhanced_menu1_runner()
        if runner:
            print("✅ Enhanced Menu 1 runner created")
        
        # Test patching
        patch_existing_menu1_implementations()
        
        print("\n🎉 Integration system ready!")
        print("📋 Available functions:")
        print("  - auto_integrate_menu1_modules()")
        print("  - create_enhanced_menu1_runner()")
        print("  - patch_existing_menu1_implementations()")
        
    else:
        print("❌ Enterprise logger not available")
        print("Please ensure enterprise_menu1_terminal_logger.py is properly installed")
