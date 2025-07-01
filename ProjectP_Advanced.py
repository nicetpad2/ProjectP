#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - MAIN ENTRY POINT
Advanced Version with Enterprise Logging & Process Tracking

Version: 3.0 Advanced Edition
Date: July 1, 2025
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import advanced logger first
from core.advanced_logger import get_advanced_logger

# Import core systems
from core.project_paths import get_project_paths
from core.config import load_config
from core.compliance import validate_enterprise_compliance


class NicegoldProjectPAdvanced:
    """Main NICEGOLD ProjectP Application - Advanced Version"""
    
    def __init__(self):
        # Initialize advanced logger
        self.logger = get_advanced_logger("NICEGOLD_MAIN")
        
        # Session info
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.version = "3.0 Advanced Edition"
        
        # Initialize components
        self.config = None
        self.paths = None
        self.menu_1 = None
        
        self.logger.info(f"🚀 NICEGOLD ProjectP Advanced starting (Session: {self.session_id})")
    
    def initialize_system(self) -> bool:
        """Initialize the system components"""
        try:
            self.logger.info("🔧 Initializing system components...")
            
            # Load configuration
            self.config = load_config()
            if not self.config:
                self.logger.warning("Configuration not loaded, using defaults")
                self.config = {}
            
            # Get project paths
            self.paths = get_project_paths()
            if not self.paths:
                raise Exception("Failed to initialize project paths")
            
            # Validate enterprise compliance
            compliance_result = validate_enterprise_compliance()
            if not compliance_result.get('compliant', False):
                self.logger.warning("Enterprise compliance validation failed")
            
            self.logger.success("✅ System initialization completed")
            return True
            
        except Exception as e:
            self.logger.critical(f"System initialization failed: {str(e)}", exception=e)
            return False
    
    def display_main_menu(self):
        """Display the main menu"""
        menu_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║              🏢 NICEGOLD ENTERPRISE PROJECTP - {self.version:<20} ║
║                     Advanced AI Trading System                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  🌊 1. Full Pipeline (Elliott Wave CNN-LSTM + DQN) - ADVANCED     ⭐ PRIMARY    ║
║  📊 2. Data Analysis & Preprocessing                   [Development]             ║
║  🤖 3. Model Training & Optimization                   [Development]             ║
║  🎯 4. Strategy Backtesting                            [Development]             ║
║  📈 5. Performance Analytics                           [Development]             ║
║                                                                                  ║
║  🔧 D. Display System Information                                               ║
║  📋 L. View Logs                                                                ║
║  🚪 E. Exit System                                                              ║
║  🔄 R. Reset & Restart                                                          ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  📊 Session: {self.session_id:<60} ║
║  ⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<64} ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
        print(menu_text)
    
    def run_menu_1(self) -> bool:
        """Run Menu 1: Elliott Wave System"""
        try:
            self.logger.info("🌊 Starting Menu 1: Elliott Wave System")
            
            # Import Menu 1 Advanced
            from menu_modules.menu_1_elliott_wave_advanced import Menu1ElliottWaveAdvanced
            
            # Create and run Menu 1
            self.menu_1 = Menu1ElliottWaveAdvanced(self.config)
            
            # Execute the full pipeline
            results = self.menu_1.run_full_pipeline()
            
            # Check results
            if results.get('status') == 'cancelled':
                self.logger.info("Menu 1 execution cancelled by user")
                return True
            elif results.get('status') == 'failed':
                self.logger.error(f"Menu 1 execution failed: {results.get('error', 'Unknown error')}")
                return False
            else:
                self.logger.success("Menu 1 execution completed successfully")
                return True
                
        except Exception as e:
            self.logger.critical(f"Menu 1 execution failed: {str(e)}", exception=e)
            return False
    
    def display_system_info(self):
        """Display system information"""
        info = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                            🔧 SYSTEM INFORMATION                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Version: {self.version:<65} ║
║  Session ID: {self.session_id:<60} ║
║  Python Version: {sys.version.split()[0]:<55} ║
║  Platform: {sys.platform:<63} ║
║                                                                                  ║
║  📁 Project Root: {str(project_root):<57} ║
║  📊 Config Loaded: {'Yes' if self.config else 'No':<58} ║
║  🛤️ Paths Initialized: {'Yes' if self.paths else 'No':<50} ║
║                                                                                  ║
║  🧠 Available Components:                                                        ║
║    • Advanced Logger with Progress Tracking                                     ║
║    • Elliott Wave CNN-LSTM Engine                                               ║
║    • DQN Reinforcement Learning Agent                                           ║
║    • SHAP + Optuna Feature Selection                                            ║
║    • Enterprise ML Protection System                                            ║
║    • Performance Analysis & Reporting                                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
        print(info)
    
    def view_logs(self):
        """View recent logs"""
        try:
            self.logger.info("📋 Displaying recent logs...")
            
            # Display performance summary
            self.logger.display_performance_summary()
            
            # Show log file locations
            print("\n📁 Log File Locations:")
            print(f"  Main Logs: logs/nicegold_advanced_{datetime.now().strftime('%Y%m%d')}.log")
            print(f"  Error Logs: logs/errors/")
            print(f"  Performance Logs: logs/performance/")
            
        except Exception as e:
            self.logger.error(f"Failed to view logs: {str(e)}", exception=e)
    
    def run_main_loop(self):
        """Run the main application loop"""
        self.logger.info("🎮 Starting main application loop")
        
        while True:
            try:
                # Display main menu
                self.display_main_menu()
                
                # Get user choice
                choice = input("\n🎯 Enter your choice: ").strip().upper()
                
                if choice == '1':
                    self.logger.info("User selected Menu 1: Elliott Wave System")
                    self.run_menu_1()
                    
                elif choice == '2':
                    self.logger.info("User selected Menu 2: Data Analysis (Development)")
                    print("\n⚠️ This menu is under development.")
                    input("Press Enter to continue...")
                    
                elif choice == '3':
                    self.logger.info("User selected Menu 3: Model Training (Development)")
                    print("\n⚠️ This menu is under development.")
                    input("Press Enter to continue...")
                    
                elif choice == '4':
                    self.logger.info("User selected Menu 4: Strategy Backtesting (Development)")
                    print("\n⚠️ This menu is under development.")
                    input("Press Enter to continue...")
                    
                elif choice == '5':
                    self.logger.info("User selected Menu 5: Performance Analytics (Development)")
                    print("\n⚠️ This menu is under development.")
                    input("Press Enter to continue...")
                    
                elif choice == 'D':
                    self.logger.info("User requested system information")
                    self.display_system_info()
                    input("Press Enter to continue...")
                    
                elif choice == 'L':
                    self.logger.info("User requested log view")
                    self.view_logs()
                    input("Press Enter to continue...")
                    
                elif choice == 'E':
                    self.logger.info("User requested system exit")
                    self.shutdown_system()
                    break
                    
                elif choice == 'R':
                    self.logger.info("User requested system restart")
                    self.restart_system()
                    
                else:
                    self.logger.warning(f"Invalid menu choice: {choice}")
                    print(f"\n❌ Invalid choice: {choice}")
                    input("Press Enter to continue...")
                    
            except KeyboardInterrupt:
                self.logger.info("User interrupted with Ctrl+C")
                print("\n\n🛑 Interrupt received. Shutting down...")
                self.shutdown_system()
                break
                
            except Exception as e:
                self.logger.error(f"Main loop error: {str(e)}", exception=e)
                print(f"\n❌ An error occurred: {str(e)}")
                input("Press Enter to continue...")
    
    def restart_system(self):
        """Restart the system"""
        try:
            self.logger.info("🔄 Restarting system...")
            
            # Save current session report
            self.logger.save_session_report()
            
            # Reinitialize
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger.info(f"🚀 System restarted (New Session: {self.session_id})")
            
            # Reinitialize components
            self.initialize_system()
            
        except Exception as e:
            self.logger.error(f"System restart failed: {str(e)}", exception=e)
    
    def shutdown_system(self):
        """Shutdown the system gracefully"""
        try:
            self.logger.info("🛑 Shutting down system...")
            
            # Save session report
            self.logger.save_session_report()
            
            # Display final summary
            print("\n" + "="*80)
            print("🎉 NICEGOLD ENTERPRISE PROJECTP - SHUTDOWN COMPLETE")
            print("="*80)
            print(f"Session ID: {self.session_id}")
            print(f"Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Thank you for using NICEGOLD Enterprise ProjectP!")
            print("="*80)
            
            self.logger.success("System shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"System shutdown error: {str(e)}", exception=e)


def main():
    """Main entry point"""
    try:
        # Create and run the application
        app = NicegoldProjectPAdvanced()
        
        # Initialize system
        if app.initialize_system():
            # Run main loop
            app.run_main_loop()
        else:
            print("❌ System initialization failed. Check logs for details.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        print("Check logs for detailed error information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
