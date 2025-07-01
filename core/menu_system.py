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

class MenuSystem:
    """ระบบเมนูหลักของ NICEGOLD Enterprise"""
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.running = True
        
        # Import Menu Modules
        self._import_menu_modules()
    
    def _import_menu_modules(self):
        """นำเข้าโมดูลเมนูต่างๆ"""
        self.menu_1 = None
        self.menu_errors = []
        
        # Try to import Menu 1 with detailed error handling
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            self.menu_1 = Menu1ElliottWave(self.config, self.logger)
            self.logger.info("✅ Menu 1 Elliott Wave Module Loaded")
            
        except ImportError as e:
            error_msg = f"Unable to import required dependencies:\n{str(e)}"
            self.menu_errors.append(("Menu 1", error_msg))
            self.logger.error(f"❌ Failed to import Menu 1: {error_msg}")
            
        except Exception as e:
            error_msg = f"Menu initialization error: {str(e)}"
            self.menu_errors.append(("Menu 1", error_msg))
            self.logger.error(f"❌ Menu 1 initialization failed: {error_msg}")
        
        # Show warning if some menus failed to load
        if self.menu_errors:
            print("⚠️ Warning: Some menu modules could not be imported:", end=" ")
            for menu_name, error in self.menu_errors:
                print(error.split('\n')[0])  # Show first line of error
    
    def display_main_menu(self):
        """แสดงเมนูหลัก"""
        print("\n" + "="*80)
        print("🏢 NICEGOLD ENTERPRISE PROJECTP - DIVINE EDITION")
        print("   AI-Powered Algorithmic Trading System")
        print("="*80)
        print()
        print("📋 MAIN MENU:")
        
        # Show Menu 1 status based on availability
        if self.menu_1:
            print("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN)")
        else:
            print("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN) [DISABLED - Dependencies missing]")
            
        print("  2. 📊 Data Analysis & Preprocessing [Under Development]")
        print("  3. 🤖 Model Training & Optimization [Under Development]")
        print("  4. 🎯 Strategy Backtesting [Under Development]")
        print("  5. 📈 Performance Analytics [Under Development]")
        print("  D. 🔧 Dependency Check & Fix")
        print("  E. 🚪 Exit System")
        print("  R. 🔄 Reset & Restart")
        print()
        
        # Show dependency warnings if any
        if self.menu_errors:
            print("⚠️  DEPENDENCY ISSUES:")
            for menu_name, error in self.menu_errors:
                print(f"    {menu_name}: Dependencies missing")
            print("    💡 Try option 'D' to fix dependencies")
            print()
            
        print("="*80)
    
    def get_user_choice(self) -> str:
        """รับข้อมูลจากผู้ใช้"""
        try:
            choice = input("🎯 Select option (1-5, D, E, R): ").strip().upper()
            return choice
        except (EOFError, KeyboardInterrupt):
            return 'E'
    
    def handle_menu_choice(self, choice: str) -> bool:
        """จัดการตัวเลือกเมนู"""
        try:
            if choice == '1':
                # Check if Menu 1 is available
                if not self.menu_1:
                    print("❌ Menu 1 is not available due to missing dependencies")
                    print("💡 Please try option 'D' to fix dependencies")
                    input("Press Enter to continue...")
                    return True
                
                self.logger.info("🌊 Starting Elliott Wave Full Pipeline...")
                
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
                
            elif choice == '2':
                print("📊 Data Analysis & Preprocessing")
                print("⚠️  Feature under development")
                input("Press Enter to continue...")
                return True
                
            elif choice == '3':
                print("🤖 Model Training & Optimization")
                print("⚠️  Feature under development")
                input("Press Enter to continue...")
                return True
                
            elif choice == '4':
                print("🎯 Strategy Backtesting")
                print("⚠️  Feature under development")
                input("Press Enter to continue...")
                return True
                
            elif choice == '5':
                print("📈 Performance Analytics")
                print("⚠️  Feature under development")
                input("Press Enter to continue...")
                return True
                
            elif choice == 'D':
                print("🔧 Dependency Check & Fix")
                self._run_dependency_fix()
                return True
                
            elif choice == 'E':
                print("🚪 Exiting NICEGOLD Enterprise System...")
                self.running = False
                return False
                
            elif choice == 'R':
                print("🔄 Restarting System...")
                return True
                
            else:
                print(f"❌ Invalid option: {choice}")
                print("Please select 1-5, D, E, or R")
                input("Press Enter to continue...")
                return True
                
        except Exception as e:
            self.logger.error(f"💥 Menu error: {str(e)}")
            print(f"❌ Error: {str(e)}")
            input("Press Enter to continue...")
            return True
    
    def _run_dependency_fix(self):
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
