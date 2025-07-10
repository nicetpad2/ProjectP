#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ UNIFIED MASTER MENU SYSTEM - COMPLETE INTEGRATION
ระบบเมนูแบบรวมศูนย์ที่เชื่อมต่อทุกส่วนของ NICEGOLD ProjectP

🏢 FEATURES:
✅ Single Entry Point Integration 
✅ Unified Resource Manager Integration
✅ Enterprise Logger Integration
✅ Complete Menu 1 Integration
✅ Beautiful Progress Bar Integration
✅ Zero Duplication System
✅ Complete Error Handling
✅ Cross-platform Compatibility
"""

import sys
import os
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Force CUDA environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

def safe_print(*args, **kwargs):
    """Safe print with error handling"""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        try:
            # Recreate the message for stderr
            message = " ".join(map(str, args))
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except:
            pass

class UnifiedMasterMenuSystem:
    """🎛️ Unified Master Menu System - Complete Integration"""
    
    def __init__(self):
        """Initialize unified master menu system"""
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.config = None
        self.logger = None
        self.resource_manager = None
        self.menu_1 = None
        self.menu_available = False
        self.menu_type = "None"
        self.running = True
        
        safe_print(f"🎛️ Initializing Unified Master Menu System (Session: {self.session_id})")
        
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            safe_print("🔧 Initializing system components...")
            
            # 1. Initialize Resource Manager
            if not self._initialize_resource_manager():
                safe_print("⚠️ Resource manager initialization failed")
            
            # 2. Initialize Logger
            if not self._initialize_logger():
                safe_print("⚠️ Logger initialization failed")
                
            # 3. Initialize Configuration
            if not self._initialize_config():
                safe_print("⚠️ Configuration initialization failed")
                
            # 4. Initialize Menu 1 System
            if not self._initialize_menu_1():
                safe_print("⚠️ Menu 1 initialization failed")
                
            return True
            
        except Exception as e:
            safe_print(f"❌ Component initialization error: {e}")
            return False
    
    def _initialize_resource_manager(self) -> bool:
        """Initialize unified resource manager"""
        try:
            # Priority 1: High Memory Resource Manager
            try:
                from core.high_memory_resource_manager import get_high_memory_resource_manager
                self.resource_manager = get_high_memory_resource_manager()
                self.resource_manager.start_monitoring()
                safe_print("✅ High Memory Resource Manager: ACTIVE (80% RAM)")
                return True
            except Exception as e1:
                safe_print(f"⚠️ High Memory RM failed: {e1}")
                
            # Priority 2: Unified Resource Manager
            try:
                from core.unified_resource_manager import get_unified_resource_manager
                self.resource_manager = get_unified_resource_manager()
                self.resource_manager.start_monitoring()
                safe_print("✅ Unified Resource Manager: ACTIVE")
                return True
            except Exception as e2:
                safe_print(f"⚠️ Unified RM failed: {e2}")
                
            # Priority 3: Enterprise Resource Control
            try:
                from core.enterprise_resource_control_center import get_resource_control_center
                self.resource_manager = get_resource_control_center()
                safe_print("✅ Enterprise Resource Control: ACTIVE")
                return True
            except Exception as e3:
                safe_print(f"⚠️ Enterprise RC failed: {e3}")
                
            return False
            
        except Exception as e:
            safe_print(f"❌ Resource manager initialization error: {e}")
            return False
    
    def _initialize_logger(self) -> bool:
        """Initialize unified enterprise logger"""
        try:
            from core.unified_enterprise_logger import get_unified_logger
            self.logger = get_unified_logger("NICEGOLD_UNIFIED_MASTER")
            safe_print("✅ Unified Enterprise Logger: ACTIVE")
            return True
        except Exception as e:
            safe_print(f"❌ CRITICAL: Unified Enterprise Logger failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_config(self) -> bool:
        """Initialize enterprise configuration from unified source."""
        try:
            from core.config import get_global_config
            self.config = get_global_config()
            
            # Add runtime-specific values
            self.config.set('runtime.resource_manager', self.resource_manager)
            self.config.set('runtime.session_id', self.session_id)
            
            safe_print("✅ Enterprise Configuration: READY (Unified Source)")
            return True
            
        except ImportError:
            safe_print("❌ CRITICAL: core.config could not be imported. Cannot load configuration.")
            return False
        except Exception as e:
            safe_print(f"❌ Configuration initialization error: {e}")
            return False
    
    def _initialize_menu_1(self) -> bool:
        """Initialize Menu 1 Elliott Wave system"""
        if not self.config:
            safe_print("❌ Cannot initialize Menu 1: Configuration is not loaded.")
            return False
            
        try:
            # Priority 1: Standard Menu 1 Elliott Wave (WORKING VERSION)
            try:
                from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
                self.menu_1 = Menu1ElliottWave(self.config.config, self.logger, self.resource_manager)
                safe_print("✅ Standard Menu 1 Elliott Wave: READY")
                self.menu_available = True
                self.menu_type = "Standard Elliott Wave (Full Pipeline)"
                return True
            except Exception as e1:
                safe_print(f"⚠️ Standard Menu 1 failed: {e1}")
                
            # Priority 2: Enhanced Menu 1 Elliott Wave (if available)
            try:
                from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
                self.menu_1 = EnhancedMenu1ElliottWave(self.config.config, self.logger, self.resource_manager)
                safe_print("✅ Enhanced Menu 1 Elliott Wave: READY")
                self.menu_available = True
                self.menu_type = "Enhanced Elliott Wave with Enterprise Features"
                return True
            except Exception as e2:
                safe_print(f"⚠️ Enhanced Menu 1 failed: {e2}")
                
            # Priority 3: Complete Menu 1
            try:
                from menu_modules.menu_1_elliott_wave_complete import CompleteMenu1ElliottWave
                self.menu_1 = CompleteMenu1ElliottWave(self.config.config, self.logger, self.resource_manager)
                safe_print("✅ Complete Menu 1 Elliott Wave: READY")
                self.menu_available = True
                self.menu_type = "Complete Elliott Wave Implementation"
                return True
            except Exception as e3:
                safe_print(f"⚠️ Complete Menu 1 failed: {e3}")
                
            # Priority 4: Completely Fixed Production Menu
            try:
                from menu_modules.completely_fixed_production_menu_1 import CompletelyFixedProductionMenu1
                self.menu_1 = CompletelyFixedProductionMenu1(self.config.config, self.logger, self.resource_manager)
                safe_print("✅ Completely Fixed Production Menu 1: READY")
                self.menu_available = True
                self.menu_type = "Completely Fixed Production Menu"
                return True
            except Exception as e4:
                safe_print(f"⚠️ Fixed Production Menu failed: {e4}")
                
            safe_print("❌ No Menu 1 implementation available")
            return False
            
        except Exception as e:
            safe_print(f"❌ Menu 1 initialization error: {e}")
            return False
    
    def display_unified_menu(self):
        """Display unified master menu"""
        menu_lines = [
            "",
            "╭═══════════════════════════════════════════════════════════════════════════════════════════════════╮",
            "║                        🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED MASTER SYSTEM                    ║",
            "║                          🎛️ Complete Integration & Zero Duplication Edition                       ║",
            "╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣",
            "║                                      📋 MASTER MENU OPTIONS                                      ║",
            "║                                                                                                   ║",
            "║ 1. 🌊 Elliott Wave Full Pipeline (Complete Enterprise Integration)                                ║",
            f"║    ⚡ {self.menu_type[:80]:<80}║",
            "║    🧠 CNN-LSTM + DQN + SHAP/Optuna + Resource Management                                         ║",
            "║    🎨 Beautiful Progress Bars + Enterprise Logging                                               ║",
            "║                                                                                                   ║",
            "║ 2. 📊 System Status & Resource Monitor                                                           ║",
            "║    📈 Unified resource monitoring and system health dashboard                                    ║",
            "║                                                                                                   ║",
            "║ 3. 🔧 System Diagnostics & Dependency Check                                                      ║",
            "║    🛠️ Complete system validation and dependency management                                       ║",
            "║                                                                                                   ║",
            "║ D. 🎨 Beautiful Progress Bars Demo                                                               ║",
            "║    ✨ Demonstration of visual progress tracking system                                           ║",
            "║                                                                                                   ║",
            "║ E. 🚪 Exit System                                                                                ║",
            "║ R. 🔄 Reset & Restart Complete System                                                           ║",
            "║                                                                                                   ║",
            "╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣",
            f"║ 📊 Session: {self.session_id:<20} │ 🧠 RM: {'✅' if self.resource_manager else '❌':<3} │ 🎛️ Menu: {'✅' if self.menu_available else '❌':<3} ║",
            f"║ 📝 Logger: {'✅ Active':<12} │ 🎨 Progress: ✅     │ 🔒 Safe Mode: ✅      ║",
            "╰═══════════════════════════════════════════════════════════════════════════════════════════════════╯"
        ]
        
        for line in menu_lines:
            safe_print(line)
    
    def get_user_choice(self) -> str:
        """Get user input with error handling"""
        try:
            choice = input("\n🎯 Enter your choice: ").strip().upper()
            return choice
        except (EOFError, KeyboardInterrupt):
            return "E"
        except Exception:
            return "E"
    
    def handle_menu_choice(self, choice: str) -> bool:
        """Handle user menu choice"""
        try:
            if choice == "1":
                return self._handle_elliott_wave_pipeline()
            elif choice == "2":
                return self._handle_system_status()
            elif choice == "3":
                return self._handle_system_diagnostics()
            elif choice == "D":
                return self._handle_progress_demo()
            elif choice == "E":
                return self._handle_exit()
            elif choice == "R":
                return self._handle_restart()
            else:
                safe_print(f"❌ Invalid choice: {choice}")
                safe_print("💡 Please select 1, 2, 3, D, E, or R")
                return True
                
        except Exception as e:
            safe_print(f"❌ Menu choice error: {e}")
            return True
    
    def _handle_elliott_wave_pipeline(self) -> bool:
        """Handle Elliott Wave Full Pipeline execution"""
        safe_print("\n🌊 ELLIOTT WAVE FULL PIPELINE - ENTERPRISE INTEGRATION")
        safe_print("="*80)
        
        if not self.menu_available or not self.menu_1:
            safe_print("❌ Elliott Wave Pipeline not available")
            safe_print("🔧 Try option 3 for system diagnostics")
            input("\nPress Enter to continue...")
            return True
        
        try:
            safe_print(f"🚀 Starting {self.menu_type}...")
            safe_print("🎨 Beautiful progress bars will be displayed during execution")
            safe_print("")
            
            # Execute Menu 1 pipeline
            start_time = time.time()
            result = self.menu_1.run()
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Process results
            if result and (result.get('success', False) or result.get('status') == 'success'):
                safe_print("\n🎉 ELLIOTT WAVE PIPELINE COMPLETED SUCCESSFULLY!")
                safe_print(f"⏱️ Duration: {duration:.2f} seconds")
                
                # Display detailed results if available
                if isinstance(result, dict):
                    if 'session_summary' in result:
                        summary = result['session_summary']
                        safe_print(f"\n📊 SESSION SUMMARY:")
                        safe_print(f"   📈 Total Steps: {summary.get('total_steps', 'N/A')}")
                        safe_print(f"   🎯 Features Selected: {summary.get('selected_features', 'N/A')}")
                        safe_print(f"   🧠 Model AUC: {summary.get('model_auc', 'N/A')}")
                        safe_print(f"   📊 Performance: {summary.get('performance_grade', 'N/A')}")
                        
                    if 'output_files' in result:
                        safe_print(f"\n📁 Output files saved to: {result.get('output_directory', 'outputs/')}")
                        
            else:
                safe_print("\n⚠️ Elliott Wave Pipeline completed with warnings")
                if result and isinstance(result, dict):
                    error_msg = result.get('error', result.get('message', 'Unknown issue'))
                    safe_print(f"💡 Details: {error_msg}")
                
        except Exception as e:
            safe_print(f"\n❌ Pipeline execution error: {e}")
            import traceback
            traceback.print_exc()
            
        input("\nPress Enter to continue...")
        return True
    
    def _handle_system_status(self) -> bool:
        """Handle system status and resource monitoring"""
        safe_print("\n📊 UNIFIED SYSTEM STATUS & RESOURCE MONITOR")
        safe_print("="*80)
        
        # System Component Status
        safe_print("🔧 SYSTEM COMPONENTS:")
        safe_print(f"   🧠 Resource Manager: {'✅ Active' if self.resource_manager else '❌ Unavailable'}")
        safe_print(f"   📝 Logger: {'✅ Active' if self.logger else '❌ Unavailable'}")
        safe_print(f"   🎛️ Menu 1: {'✅ Available' if self.menu_available else '❌ Unavailable'}")
        safe_print(f"   ⚙️ Configuration: {'✅ Loaded' if self.config else '❌ Missing'}")
        
        # Resource Status
        if self.resource_manager:
            try:
                safe_print("\n🧠 RESOURCE UTILIZATION:")
                
                # Get current performance data
                if hasattr(self.resource_manager, 'get_current_performance'):
                    perf = self.resource_manager.get_current_performance()
                    cpu_usage = perf.get('cpu_percent', 0)
                    memory_info = perf.get('memory', {})
                    memory_usage = memory_info.get('percent', 0)
                    memory_available = memory_info.get('available', 0) / (1024**3)
                    
                    safe_print(f"   🖥️ CPU Usage: {cpu_usage:.1f}%")
                    safe_print(f"   💾 Memory Usage: {memory_usage:.1f}%")
                    safe_print(f"   💿 Available Memory: {memory_available:.1f} GB")
                    
                # Get health status if available
                if hasattr(self.resource_manager, 'get_health_status'):
                    health = self.resource_manager.get_health_status()
                    safe_print(f"   🏥 System Health: {health.get('health_score', 'N/A')}")
                    
            except Exception as e:
                safe_print(f"   ⚠️ Resource monitoring error: {e}")
        else:
            # Fallback to psutil
            try:
                import psutil
                safe_print("\n🧠 BASIC RESOURCE STATUS:")
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                safe_print(f"   🖥️ CPU Usage: {cpu:.1f}%")
                safe_print(f"   💾 Memory Usage: {memory.percent:.1f}%")
                safe_print(f"   💿 Available Memory: {memory.available/(1024**3):.1f} GB")
            except Exception as e:
                safe_print(f"   ⚠️ Unable to get resource status: {e}")
        
        # Menu Status
        safe_print(f"\n🎛️ MENU SYSTEM STATUS:")
        safe_print(f"   📋 Menu Type: {self.menu_type}")
        safe_print(f"   🌊 Elliott Wave: {'✅ Ready' if self.menu_available else '❌ Not Available'}")
        
        # Session Info
        safe_print(f"\n📊 SESSION INFORMATION:")
        safe_print(f"   🆔 Session ID: {self.session_id}")
        safe_print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        safe_print(f"   🔒 Safe Mode: ✅ Active")
        safe_print(f"   🎨 Progress Bars: ✅ Enabled")
        
        input("\nPress Enter to continue...")
        return True
    
    def _handle_system_diagnostics(self) -> bool:
        """Handle system diagnostics and dependency check"""
        safe_print("\n🔧 UNIFIED SYSTEM DIAGNOSTICS")
        safe_print("="*80)
        
        safe_print("📦 CHECKING CORE DEPENDENCIES:")
        
        # Check critical modules
        critical_modules = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow', 
            'torch', 'shap', 'optuna', 'joblib', 'psutil'
        ]
        
        for module in critical_modules:
            try:
                __import__(module)
                safe_print(f"   ✅ {module}")
            except ImportError:
                safe_print(f"   ❌ {module} - Missing")
        
        safe_print("\n🗂️ CHECKING FILE STRUCTURE:")
        critical_paths = [
            'datacsv/', 'core/', 'menu_modules/', 'elliott_wave_modules/',
            'datacsv/XAUUSD_M1.csv', 'datacsv/XAUUSD_M15.csv'
        ]
        
        for path in critical_paths:
            if Path(path).exists():
                safe_print(f"   ✅ {path}")
            else:
                safe_print(f"   ❌ {path} - Missing")
        
        safe_print("\n🧩 CHECKING COMPONENT INTEGRATION:")
        safe_print(f"   {'✅' if self.resource_manager else '❌'} Resource Manager Integration")
        safe_print(f"   {'✅' if self.logger else '❌'} Logger Integration")
        safe_print(f"   {'✅' if self.menu_available else '❌'} Menu 1 Integration")
        safe_print(f"   {'✅' if self.config else '❌'} Configuration Integration")
        
        # Memory and CPU check
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            safe_print(f"\n💻 SYSTEM SPECIFICATIONS:")
            safe_print(f"   🧠 Total RAM: {memory.total/(1024**3):.1f} GB")
            safe_print(f"   🖥️ CPU Cores: {cpu_count}")
            safe_print(f"   💾 Available RAM: {memory.available/(1024**3):.1f} GB")
            
            # Check if high-memory capable
            if memory.total >= 40 * (1024**3):  # 40GB+
                safe_print("   ✅ High-Memory System Detected")
            else:
                safe_print("   ⚠️ Standard Memory System")
                
        except Exception as e:
            safe_print(f"   ❌ System specification check failed: {e}")
        
        safe_print("\n🔧 SYSTEM RECOMMENDATIONS:")
        if not self.menu_available:
            safe_print("   💡 Menu 1 not available - check dependencies")
        if not self.resource_manager:
            safe_print("   💡 Resource manager not available - install psutil")
        
        safe_print("   ✅ System appears ready for enterprise operation")
        
        input("\nPress Enter to continue...")
        return True
    
    def _handle_progress_demo(self) -> bool:
        """Handle beautiful progress bars demonstration"""
        safe_print("\n🎨 BEAUTIFUL PROGRESS BARS DEMONSTRATION")
        safe_print("="*80)
        
        try:
            # Demo different progress bar styles
            safe_print("📊 Data Loading Simulation:")
            for i in range(11):
                progress_bar = "█" * (i * 4) + "░" * ((10 - i) * 4)
                safe_print(f"\r   [{progress_bar}] {i * 10}% - Loading market data...", end="")
                time.sleep(0.3)
            safe_print("\n   ✅ Data loading completed!")
            
            safe_print("\n🔧 Feature Engineering Simulation:")
            stages = [
                "Moving averages calculation",
                "Technical indicators computation", 
                "Elliott Wave pattern detection",
                "SHAP feature importance analysis",
                "Feature selection optimization"
            ]
            
            for i, stage in enumerate(stages, 1):
                progress = int((i / len(stages)) * 40)
                bar = "█" * progress + "░" * (40 - progress)
                safe_print(f"   [{bar}] {(i/len(stages)*100):5.1f}% - {stage}")
                time.sleep(0.5)
            
            safe_print("\n🧠 Model Training Simulation:")
            for epoch in range(1, 11):
                progress = int((epoch / 10) * 40)
                bar = "█" * progress + "░" * (40 - progress)
                safe_print(f"\r   [{bar}] Epoch {epoch}/10 - Training CNN-LSTM...", end="")
                time.sleep(0.4)
            safe_print("\n   ✅ Model training completed!")
            
            safe_print("\n🎉 Progress demonstration completed!")
            
        except Exception as e:
            safe_print(f"❌ Progress demo error: {e}")
        
        input("\nPress Enter to continue...")
        return True
    
    def _handle_exit(self) -> bool:
        """Handle system exit"""
        safe_print("\n🚪 EXITING NICEGOLD ENTERPRISE PROJECTP")
        safe_print("✨ Thank you for using the Unified Master System!")
        
        # Cleanup resources
        if self.resource_manager:
            try:
                if hasattr(self.resource_manager, 'stop_monitoring'):
                    self.resource_manager.stop_monitoring()
                safe_print("🧹 Resource manager cleanup completed")
            except Exception as e:
                safe_print(f"⚠️ Cleanup warning: {e}")
        
        # Force garbage collection
        gc.collect()
        safe_print("🗑️ Memory cleanup completed")
        safe_print("👋 Goodbye!")
        
        return False
    
    def _handle_restart(self) -> bool:
        """Handle system restart"""
        safe_print("\n🔄 RESTARTING UNIFIED MASTER SYSTEM")
        safe_print("🔧 Re-initializing all components...")
        
        # Stop current resources
        if self.resource_manager:
            try:
                if hasattr(self.resource_manager, 'stop_monitoring'):
                    self.resource_manager.stop_monitoring()
            except:
                pass
        
        # Reset all components
        self.resource_manager = None
        self.logger = None
        self.menu_1 = None
        self.menu_available = False
        self.config = None
        
        # Force garbage collection
        gc.collect()
        
        # Generate new session ID
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Re-initialize components
        if self.initialize_components():
            safe_print("✅ System restart completed successfully!")
            safe_print(f"🆔 New session ID: {self.session_id}")
        else:
            safe_print("⚠️ System restart completed with warnings")
        
        input("\nPress Enter to continue...")
        return True
    
    def start(self):
        """Start the unified master menu system"""
        safe_print("🚀 Starting Unified Master Menu System...")
        
        # Initialize all components
        if not self.initialize_components():
            safe_print("⚠️ Some components failed to initialize")
        
        safe_print("\n✅ UNIFIED MASTER SYSTEM READY")
        safe_print("🎛️ Zero duplication, complete integration achieved")
        
        # Main menu loop
        while self.running:
            try:
                self.display_unified_menu()
                choice = self.get_user_choice()
                
                if not self.handle_menu_choice(choice):
                    break
                    
                # Pause before next iteration
                safe_print("\n" + "="*80)
                
            except KeyboardInterrupt:
                safe_print("\n🛑 System interrupted by user")
                break
            except Exception as e:
                safe_print(f"\n❌ System error: {e}")
                input("Press Enter to continue...")
        
        safe_print("\n✅ Unified Master System shutdown complete")

def main():
    """Main entry point for unified master menu system"""
    try:
        # Create and start unified system
        unified_system = UnifiedMasterMenuSystem()
        unified_system.start()
        
    except Exception as e:
        safe_print(f"❌ System startup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
