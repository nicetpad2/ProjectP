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
                return self.menu_1.execute_full_pipeline()
                
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
