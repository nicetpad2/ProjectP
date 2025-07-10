#!/usr/bin/env python3
"""
🎛️ OPTIMIZED MENU SYSTEM
High-performance menu system optimized for resource efficiency
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import gc

# Import optimized components
try:
    from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    from core.cuda_elimination import suppress_all_cuda_output
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

class OptimizedMenuSystem:
    """Optimized menu system for high-performance operations"""
    
    def __init__(self, config: Dict = None, logger=None, resource_manager=None, progress_manager=None):
        self.config = config or {}
        self.resource_manager = resource_manager
        self.progress_manager = progress_manager
        self.running = True
        
        # Initialize optimized logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_unified_logger()
            if not self.progress_manager:
                self.progress_manager = get_progress_manager()
            self.logger.system("🎛️ Optimized Menu System initialized", component="Optimized_Menu")
        else:
            self.logger = logger or get_unified_logger()
        
        # Import optimized menu modules
        self._import_optimized_menu_modules()
    
    def _import_optimized_menu_modules(self):
        """Import optimized menu modules with error handling"""
        self.menu_1 = None
        self.menu_errors = []
        
        # Import optimized Menu 1
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                import_progress = self.progress_manager.create_progress(
                    "📋 Loading Optimized Menu Modules", 2, ProgressType.PROCESSING
                )
                self.progress_manager.update_progress(import_progress, 1, "Loading Optimized Menu 1")
            
            with suppress_all_cuda_output():
                from menu_modules.optimized_menu_1_elliott_wave import OptimizedMenu1ElliottWave
            
            self.menu_1 = OptimizedMenu1ElliottWave(
                self.config, 
                self.logger, 
                self.resource_manager
            )
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.success("✅ Optimized Menu 1 Elliott Wave Loaded", component="Menu_Import")
                self.progress_manager.update_progress(import_progress, 1, "Optimized Menu 1 loaded")
                self.progress_manager.complete_progress(import_progress, "✅ Optimized menus loaded")
            else:
                self.logger.info("✅ Optimized Menu 1 Elliott Wave Loaded")
            
            # Memory optimization after loading
            gc.collect()
            
        except ImportError as e:
            # Fallback to standard menu with optimization
            try:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.warning("Optimized Menu 1 unavailable, loading standard with optimization", component="Menu_Import")
                
                with suppress_all_cuda_output():
                    from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
                
                self.menu_1 = Menu1ElliottWave(self.config, self.logger, self.resource_manager)
                
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("✅ Standard Menu 1 loaded with optimization", component="Menu_Import")
                
            except Exception as fallback_e:
                error_msg = f"Menu 1 loading failed: {str(fallback_e)}"
                self.menu_errors.append(("Menu 1", error_msg))
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.error("❌ All Menu 1 loading attempts failed", "Menu_Import", 
                                    data={'error': error_msg}, exception=fallback_e)
        
        except Exception as e:
            error_msg = f"Optimized Menu 1 initialization error: {str(e)}"
            self.menu_errors.append(("Optimized Menu 1", error_msg))
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error("❌ Optimized Menu 1 failed", "Menu_Import", 
                                data={'error': error_msg}, exception=e)
    
    def start(self):
        """Start optimized menu system"""
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system("🎛️ Starting Optimized Menu System", component="Menu_Start")
        
        while self.running:
            try:
                self._display_optimized_main_menu()
                choice = self._get_optimized_user_input()
                self._handle_optimized_menu_choice(choice)
                
                # Memory optimization after each menu operation
                gc.collect()
                
            except KeyboardInterrupt:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.info("🛑 Menu interrupted by user", component="Menu_System")
                self.running = False
            except Exception as e:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.error(f"Menu error: {str(e)}", "Menu_System", exception=e)
                else:
                    self.logger.error(f"Menu error: {str(e)}")
    
    def _display_optimized_main_menu(self):
        """Display optimized main menu"""
        print("\n" + "="*80)
        print("🏢 NICEGOLD ENTERPRISE - OPTIMIZED MENU SYSTEM")
        print("="*80)
        
        # Show resource status if available
        if self.resource_manager:
            try:
                status = self.resource_manager.get_health_status()
                print(f"📊 System Health: CPU {status.get('cpu_usage', 0):.1f}% | Memory {status.get('memory_usage', 0):.1f}%")
            except:
                pass
        
        print("\n🎯 Available Options:")
        
        if self.menu_1:
            print("1. 🌊 Elliott Wave Full Pipeline (Optimized)")
        else:
            print("1. ❌ Elliott Wave Pipeline (Unavailable)")
        
        print("2. 📊 System Resource Monitor")
        print("3. 🔧 System Optimization Tools")
        print("4. 📋 System Health Report")
        print("0. 🚪 Exit")
        
        if self.menu_errors:
            print(f"\n⚠️ {len(self.menu_errors)} menu(s) have loading errors")
        
        print("="*80)
    
    def _get_optimized_user_input(self):
        """Get user input with optimization"""
        try:
            choice = input("\n🎯 Select option (0-4): ").strip()
            return choice
        except (EOFError, KeyboardInterrupt):
            return "0"
    
    def _handle_optimized_menu_choice(self, choice: str):
        """Handle menu choice with optimization"""
        if choice == "1":
            if self.menu_1:
                try:
                    if ADVANCED_LOGGING_AVAILABLE:
                        self.logger.system("🌊 Starting Optimized Elliott Wave Pipeline", component="Menu_Choice")
                    
                    self.menu_1.run()
                    
                    # Memory cleanup after pipeline
                    gc.collect()
                    
                except Exception as e:
                    if ADVANCED_LOGGING_AVAILABLE:
                        self.logger.error(f"Elliott Wave Pipeline error: {str(e)}", "Menu_Choice", exception=e)
                    else:
                        print(f"❌ Elliott Wave error: {e}")
            else:
                print("❌ Elliott Wave Pipeline is not available")
        
        elif choice == "2":
            self._show_resource_monitor()
        
        elif choice == "3":
            self._show_optimization_tools()
        
        elif choice == "4":
            self._show_health_report()
        
        elif choice == "0":
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info("👋 User selected exit", component="Menu_Choice")
            print("👋 Goodbye!")
            self.running = False
        
        else:
            print("❌ Invalid choice. Please select 0-4.")
    
    def _show_resource_monitor(self):
        """Show resource monitoring information"""
        print("\n📊 SYSTEM RESOURCE MONITOR")
        print("="*50)
        
        if self.resource_manager:
            try:
                status = self.resource_manager.get_detailed_status()
                for key, value in status.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"❌ Resource monitor error: {e}")
        else:
            import psutil
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            print(f"CPU Usage: {cpu}%")
            print(f"Memory Usage: {memory.percent}%")
            print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
        
        input("\nPress Enter to continue...")
    
    def _show_optimization_tools(self):
        """Show optimization tools"""
        print("\n🔧 SYSTEM OPTIMIZATION TOOLS")
        print("="*50)
        print("1. 🧹 Run Garbage Collection")
        print("2. 🔄 Reset Resource Manager")
        print("3. 📊 Memory Usage Analysis") 
        print("0. 🔙 Back to Main Menu")
        
        choice = input("\nSelect tool (0-3): ").strip()
        
        if choice == "1":
            print("🧹 Running garbage collection...")
            collected = gc.collect()
            print(f"✅ Collected {collected} objects")
        
        elif choice == "2":
            if self.resource_manager:
                try:
                    self.resource_manager.reset_monitoring()
                    print("✅ Resource manager reset")
                except:
                    print("❌ Reset failed")
        
        elif choice == "3":
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Process Memory: {memory_info.rss / (1024**2):.1f} MB")
            print(f"Virtual Memory: {memory_info.vms / (1024**2):.1f} MB")
        
        if choice != "0":
            input("\nPress Enter to continue...")
    
    def _show_health_report(self):
        """Show comprehensive health report"""
        print("\n📋 SYSTEM HEALTH REPORT")
        print("="*50)
        
        # Show menu loading status
        print(f"🎛️ Menu Status:")
        print(f"   Menu 1 (Elliott Wave): {'✅ Available' if self.menu_1 else '❌ Unavailable'}")
        
        if self.menu_errors:
            print(f"\n⚠️ Errors ({len(self.menu_errors)}):")
            for menu, error in self.menu_errors:
                print(f"   {menu}: {error}")
        else:
            print("\n✅ No menu loading errors")
        
        # Resource status
        if self.resource_manager:
            print(f"\n🧠 Resource Manager: ✅ Active")
            try:
                status = self.resource_manager.get_health_status()
                print(f"   CPU Usage: {status.get('cpu_usage', 0):.1f}%")
                print(f"   Memory Usage: {status.get('memory_usage', 0):.1f}%")
            except:
                print("   Status: ⚠️ Monitoring unavailable")
        else:
            print(f"\n🧠 Resource Manager: ❌ Not available")
        
        print(f"\n📊 Logging: {'✅ Advanced' if ADVANCED_LOGGING_AVAILABLE else '⚠️ Basic'}")
        
        input("\nPress Enter to continue...")
