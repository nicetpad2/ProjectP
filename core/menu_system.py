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
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            self.menu_1 = Menu1ElliottWave(self.config, self.logger)
            self.logger.info("✅ Menu 1 Elliott Wave Module Loaded")
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import Menu 1: {e}")
            sys.exit(1)
    
    def display_main_menu(self):
        """แสดงเมนูหลัก"""
        print("\n" + "="*80)
        print("🏢 NICEGOLD ENTERPRISE PROJECTP - DIVINE EDITION")
        print("   AI-Powered Algorithmic Trading System")
        print("="*80)
        print()
        print("📋 MAIN MENU:")
        print("  1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN)")
        print("  2. 📊 Data Analysis & Preprocessing")
        print("  3. 🤖 Model Training & Optimization")
        print("  4. 🎯 Strategy Backtesting")
        print("  5. 📈 Performance Analytics")
        print("  E. 🚪 Exit System")
        print("  R. 🔄 Reset & Restart")
        print()
        print("="*80)
    
    def get_user_choice(self) -> str:
        """รับข้อมูลจากผู้ใช้"""
        try:
            choice = input("🎯 Select option (1-5, E, R): ").strip().upper()
            return choice
        except (EOFError, KeyboardInterrupt):
            return 'E'
    
    def handle_menu_choice(self, choice: str) -> bool:
        """จัดการตัวเลือกเมนู"""
        try:
            if choice == '1':
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
                
            elif choice == 'E':
                print("🚪 Exiting NICEGOLD Enterprise System...")
                self.running = False
                return False
                
            elif choice == 'R':
                print("🔄 Restarting System...")
                return True
                
            else:
                print(f"❌ Invalid option: {choice}")
                print("Please select 1-5, E, or R")
                input("Press Enter to continue...")
                return True
                
        except Exception as e:
            self.logger.error(f"💥 Menu error: {str(e)}")
            print(f"❌ Error: {str(e)}")
            input("Press Enter to continue...")
            return True
    
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
