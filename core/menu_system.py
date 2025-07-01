#!/usr/bin/env python3
"""
🎛️ NICEGOLD ENTERPRISE MENU SYSTEM
ระบบเมนูหลักสำหรับการจัดการระบบ Enterprise
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Import advanced logging system
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

class MenuSystem:
    """ระบบเมนูหลักของ NICEGOLD Enterprise"""
    
    def __init__(self, config: Dict = None, logger=None, resource_manager=None):
        self.config = config or {}
        self.resource_manager = resource_manager
        self.running = True
        
        # Initialize logging system
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.progress_manager = get_progress_manager()
            self.logger.system("🎛️ Menu System initialized with advanced logging", "Menu_System")
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_manager = None
        
        # Import Menu Modules
        self._import_menu_modules()
    
    def _import_menu_modules(self):
        """นำเข้าโมดูลเมนูต่างๆ"""
        self.menu_1 = None
        self.menu_errors = []
        
        # Start menu import progress
        import_progress = None
        if ADVANCED_LOGGING_AVAILABLE and self.progress_manager:
            import_progress = self.progress_manager.create_progress(
                "📋 Loading Menu Modules", 3, ProgressType.PROCESSING
            )
        
        # Try to import Menu 1 with detailed error handling
        try:
            if import_progress:
                self.progress_manager.update_progress(import_progress, 1, "Loading Menu 1 - Elliott Wave")
            
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            self.menu_1 = Menu1ElliottWave(self.config, self.logger, self.resource_manager)
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.success("✅ Menu 1 Elliott Wave Module Loaded", "Menu_Import")
            else:
                self.logger.info("✅ Menu 1 Elliott Wave Module Loaded")
            
            # If resource manager is available, show integration status
            if self.resource_manager:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.system("✅ Menu 1 integrated with Intelligent Resource Management", "Menu_Import")
                else:
                    self.logger.info("✅ Menu 1 integrated with Intelligent Resource Management")
            
        except ImportError as e:
            error_msg = f"Unable to import required dependencies:\n{str(e)}"
            self.menu_errors.append(("Menu 1", error_msg))
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Failed to import Menu 1", "Menu_Import", 
                                data={'error': error_msg}, exception=e)
            else:
                self.logger.error(f"❌ Failed to import Menu 1: {error_msg}")
            
        except Exception as e:
            error_msg = f"Menu initialization error: {str(e)}"
            self.menu_errors.append(("Menu 1", error_msg))
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Menu 1 initialization failed", "Menu_Import", 
                                data={'error': error_msg}, exception=e)
            else:
                self.logger.error(f"❌ Menu 1 initialization failed: {error_msg}")
        
        # Complete menu import
        if import_progress:
            self.progress_manager.update_progress(import_progress, 1, "Menu modules loaded")
            self.progress_manager.complete_progress(import_progress, 
                                                   f"✅ Menu loading completed: {len(self.menu_errors)} errors")
        
        # Show warning if some menus failed to load
        if self.menu_errors:
            warning_msg = "Some menu modules could not be imported"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(warning_msg, "Menu_Import", 
                                  data={'failed_menus': [error[0] for error in self.menu_errors]})
            else:
                print("⚠️ Warning: " + warning_msg + ":", end=" ")
                for menu_name, error in self.menu_errors:
                    print(error.split('\n')[0])  # Show first line of error
    
    def display_main_menu(self):
        """แสดงเมนูหลัก"""
        menu_display = """
================================================================================
🏢 NICEGOLD ENTERPRISE PROJECTP - DIVINE EDITION
   AI-Powered Algorithmic Trading System
================================================================================

📋 MAIN MENU:"""
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system(menu_display.strip(), "Menu_Display")
        else:
            print(menu_display)
        
        # Build menu options
        menu_options = []
        
        # Show Menu 1 status based on availability
        if self.menu_1:
            if self.resource_manager:
                menu_options.append("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN) ⚡ Resource Optimized")
            else:
                menu_options.append("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN)")
        else:
            menu_options.append("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN) [DISABLED - Dependencies missing]")
            
        menu_options.extend([
            "  2. 📊 Data Analysis & Preprocessing [Under Development]",
            "  3. 🤖 Model Training & Optimization [Under Development]", 
            "  4. 🎯 Strategy Backtesting [Under Development]",
            "  5. 📈 Performance Analytics [Under Development]",
            "  D. 🔧 Dependency Check & Fix",
            "  E. 🚪 Exit System",
            "  R. 🔄 Reset & Restart"
        ])
        
        # Display menu options
        for option in menu_options:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info(option.strip(), "Menu_Display")
            else:
                print(option)
        
        # Show dependency warnings if any
        if self.menu_errors:
            warning_text = [
                "",
                "⚠️  DEPENDENCY ISSUES:"
            ]
            
            for menu_name, error in self.menu_errors:
                warning_text.append(f"    {menu_name}: Dependencies missing")
            
            warning_text.extend([
                "    💡 Try option 'D' to fix dependencies",
                ""
            ])
            
            if ADVANCED_LOGGING_AVAILABLE:
                for line in warning_text:
                    if line.strip():
                        self.logger.warning(line.strip(), "Menu_Display")
            else:
                for line in warning_text:
                    print(line)
        
        # Add separator
        separator = "=" * 80
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system(separator, "Menu_Display")
        else:
            print(separator)
    
    def get_user_choice(self) -> str:
        """รับข้อมูลจากผู้ใช้"""
        try:
            choice = input("🎯 Select option (1-5, D, E, R): ").strip().upper()
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info(f"User selected option: {choice}", "User_Input")
            
            return choice
        except (EOFError, KeyboardInterrupt):
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning("User cancelled input", "User_Input")
            return 'E'

    def handle_menu_choice(self, choice: str):
        """จัดการตัวเลือกของผู้ใช้"""
        
        # Start choice handling progress
        choice_progress = None
        if ADVANCED_LOGGING_AVAILABLE and self.progress_manager:
            choice_progress = self.progress_manager.create_progress(
                f"🎯 Processing Choice: {choice}", 3, ProgressType.PROCESSING
            )
        
        try:
            if choice_progress:
                self.progress_manager.update_progress(choice_progress, 1, f"Validating choice: {choice}")
            
            if choice == '1':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("🌊 Starting Full Pipeline (Elliott Wave)", "Menu_Selection")
                
                if choice_progress:
                    self.progress_manager.update_progress(choice_progress, 1, "Preparing Elliott Wave pipeline")
                
                self._handle_menu_1()
                
            elif choice == '2':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("📊 Data Analysis & Preprocessing selected", "Menu_Selection")
                self._handle_under_development("Data Analysis & Preprocessing")
                
            elif choice == '3':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("🤖 Model Training & Optimization selected", "Menu_Selection")
                self._handle_under_development("Model Training & Optimization")
                
            elif choice == '4':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("🎯 Strategy Backtesting selected", "Menu_Selection")
                self._handle_under_development("Strategy Backtesting")
                
            elif choice == '5':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("📈 Performance Analytics selected", "Menu_Selection")
                self._handle_under_development("Performance Analytics")
                
            elif choice == 'D':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.system("🔧 Dependency Check & Fix selected", "Menu_Selection")
                self._handle_dependency_check()
                
            elif choice == 'E':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.system("🚪 Exit System selected", "Menu_Selection")
                self._handle_exit()
                
            elif choice == 'R':
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.system("🔄 Reset & Restart selected", "Menu_Selection")
                self._handle_reset()
                
            else:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.warning(f"Invalid option selected: {choice}", "Menu_Selection")
                else:
                    print(f"❌ Invalid option: {choice}")
                    print("Please select a valid option (1-5, D, E, R)")
            
            # Complete choice handling
            if choice_progress:
                self.progress_manager.complete_progress(choice_progress, f"✅ Choice {choice} processed")
                
        except Exception as e:
            if choice_progress:
                self.progress_manager.fail_progress(choice_progress, f"Choice processing failed: {str(e)}")
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Error handling menu choice {choice}", "Menu_Selection", exception=e)
            else:
                print(f"❌ Error handling choice {choice}: {str(e)}")

    def _handle_menu_1(self):
        """Handle Menu 1 actions"""
        try:
            # Check if Menu 1 is available
            if not self.menu_1:
                print("❌ Menu 1 is not available due to missing dependencies")
                print("💡 Please try option 'D' to fix dependencies")
                input("Press Enter to continue...")
                return
            
            self.logger.info("🌊 Starting Elliott Wave Full Pipeline...")
            
            # Show resource management status if available
            if self.resource_manager:
                print("⚡ Resource optimization is active during pipeline execution")
                self.logger.info("⚡ Pipeline execution with intelligent resource management")
                
                # Display current resource utilization
                current_perf = self.resource_manager.get_current_performance()
                cpu_usage = current_perf.get('cpu_percent', 0)
                memory_info = current_perf.get('memory', {})
                memory_usage = memory_info.get('percent', 0)
                
                print(f"📊 Current System Usage: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%")
                
                # Start real-time monitoring
                if not self.resource_manager.monitoring_active:
                    self.resource_manager.start_monitoring(interval=1.0)
                    print("📈 Real-time resource monitoring started")
            
            # Import Menu1Logger for enterprise-grade logging
            from core.menu1_logger import (
                start_menu1_session, 
                complete_menu1_session,
                get_menu1_logger
            )
            
            # Start enterprise logging session
            session_id = f"menu1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            menu1_logger = start_menu1_session(session_id)
            
            try:
                # Execute Menu 1 with enhanced logging
                results = self.menu_1.run_full_pipeline()
                
                if results and not results.get('error', False):
                    # Successful completion
                    final_results = {
                        "execution_status": "success",
                        "auc_score": results.get('performance_analysis', {}).get('auc_score', 0.0),
                        "enterprise_compliant": results.get('enterprise_compliance', {}).get('real_data_only', False),
                        "total_features": results.get('feature_selection', {}).get('selected_features_count', 0),
                        "pipeline_duration": results.get('execution_time', 'N/A')
                    }
                    complete_menu1_session(final_results)
                    
                    # Show resource usage summary if available
                    if self.resource_manager:
                        print("\n📊 Resource Usage Summary:")
                        
                        # Get performance data
                        performance_data = self.resource_manager.performance_data
                        if performance_data:
                            cpu_values = [d.get('cpu_percent', 0) for d in performance_data]
                            memory_values = [d.get('memory_percent', 0) for d in performance_data]
                            
                            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
                            avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
                            max_cpu = max(cpu_values) if cpu_values else 0
                            max_memory = max(memory_values) if memory_values else 0
                            
                            print(f"   🧮 CPU Usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
                            print(f"   🧠 Memory Usage - Average: {avg_memory:.1f}%, Peak: {max_memory:.1f}%")
                            print(f"   📈 Monitoring Points: {len(performance_data)} data points collected")
                            print("   ⚡ Resource optimization was active during execution")
                        
                    print("\n🎉 Elliott Wave Pipeline completed successfully!")
                    print(f"📁 Session logs saved with ID: {session_id}")
                    input("Press Enter to continue...")
                    return True
                else:
                    # Failed execution
                    error_results = {
                        "execution_status": "failed",
                        "error_message": results.get('message', 'Unknown error'),
                        "pipeline_duration": "N/A"
                    }
                    complete_menu1_session(error_results)
                    
                    print("❌ Elliott Wave Pipeline failed!")
                    print(f"📋 Check session logs: {session_id}")
                    input("Press Enter to continue...")
                    return True
                    
            except Exception as pipeline_error:
                # Critical pipeline error
                error_results = {
                    "execution_status": "critical_error",
                    "error_message": str(pipeline_error),
                    "pipeline_duration": "N/A"
                }
                complete_menu1_session(error_results)
                
                print(f"💥 Critical Pipeline Error: {str(pipeline_error)}")
                print(f"📋 Full error logs saved: {session_id}")
                input("Press Enter to continue...")
                return True
            
        except Exception as e:
            self.logger.error(f"💥 Menu error: {str(e)}")
            print(f"❌ Error: {str(e)}")
            input("Press Enter to continue...")
            return True
    
    def _handle_under_development(self, feature_name: str):
        """Handle under development features"""
        print(f"{feature_name}")
        print("⚠️  Feature under development")
        input("Press Enter to continue...")
        return True
    
    def _handle_dependency_check(self):
        """แก้ไขปัญหา dependencies"""
        import subprocess
        import sys
        
        print("🔧 NICEGOLD Dependency Fix & Check")
        print("=" * 50)
        
        # Check current NumPy status
        print("📊 Checking NumPy status...")
        try:
            import numpy as np
            print(f"✅ NumPy {np.__version__} is working")
            numpy_ok = True
        except Exception as e:
            print(f"❌ NumPy error: {e}")
            numpy_ok = False
        
        # If NumPy has issues, try to fix
        if not numpy_ok:
            print("\n🔧 Attempting to fix NumPy...")
            try:
                # Uninstall and reinstall NumPy 1.26.4
                print("   Uninstalling NumPy...")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                             capture_output=True, check=True)
                
                print("   Installing NumPy 1.26.4...")
                subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", 
                               "--no-cache-dir", "--force-reinstall"], 
                             capture_output=True, check=True)
                
                print("✅ NumPy reinstallation completed")
                
                # Test NumPy again
                try:
                    import numpy as np
                    print(f"✅ NumPy {np.__version__} is now working!")
                except Exception as e:
                    print(f"❌ NumPy still has issues: {e}")
                    
            except Exception as fix_error:
                print(f"❌ Failed to fix NumPy: {fix_error}")
        
        # Test SHAP
        print("\n📊 Checking SHAP status...")
        try:
            import shap
            print(f"✅ SHAP {shap.__version__} is working")
        except Exception as e:
            print(f"❌ SHAP error: {e}")
            
            # Try to install SHAP
            try:
                print("   Installing SHAP...")
                subprocess.run([sys.executable, "-m", "pip", "install", "shap==0.45.0"], 
                             capture_output=True, check=True)
                print("✅ SHAP installation completed")
            except Exception as shap_error:
                print(f"❌ Failed to install SHAP: {shap_error}")
        
        # Re-attempt to import Menu 1
        print("\n🔄 Re-checking Menu 1 availability...")
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            self.menu_1 = Menu1ElliottWave(self.config, self.logger)
            self.menu_errors = []  # Clear errors
            print("✅ Menu 1 is now available!")
            
        except Exception as e:
            print(f"❌ Menu 1 still unavailable: {e}")
            
        print("\n🎯 Dependency check completed!")
        print("💡 Return to main menu to try Menu 1 again")
        input("Press Enter to continue...")
    
    def _handle_exit(self):
        """Handle system exit"""
        print("🚪 Exiting NICEGOLD Enterprise System...")
        self.running = False
    
    def _handle_reset(self):
        """Handle system reset"""
        print("🔄 Restarting System...")
        self.running = True
    
    def start(self):
        """เริ่มต้นระบบเมนู"""
        self.logger.info("🎛️ Menu System Started")
        
        while self.running:
            try:
                self.display_main_menu()
                choice = self.get_user_choice()
                
                if not self.handle_menu_choice(choice):
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 System interrupted by user")
                break
                
            except Exception as e:
                self.logger.error(f"💥 System error: {str(e)}")
                print(f"❌ System error: {str(e)}")
                break
        
        self.logger.info("✅ Menu System Shutdown Complete")
