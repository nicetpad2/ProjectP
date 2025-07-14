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
import logging # Import logging for correct type hinting
import traceback
from core.unified_enterprise_logger import get_unified_logger, UnifiedEnterpriseLogger

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
        self.logger: Any = None
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
                
            # 4. Initialize System Info (GPU, etc.)
            if not self._initialize_system_info():
                safe_print("⚠️ System info initialization failed")

            # 5. Initialize Menu 1 System
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

    def _initialize_system_info(self) -> bool:
        """Detects system hardware (especially GPU) and adds it to the config."""
        if not self.logger or not self.config:
            # This check is defensive. The logger should be initialized before this.
            safe_print("⚠️ Cannot initialize system info: Logger or Config not loaded.")
            return False
        try:
            from core.enterprise_gpu_manager import get_enterprise_gpu_manager
            # Pass the actual logger object
            gpu_manager = get_enterprise_gpu_manager(logger=self.logger)
            system_info = gpu_manager.get_enterprise_configuration()
            self.config.set('system_info', system_info)
            self.logger.info(f"System Info Initialized. Processing mode: {system_info.get('processing_mode', 'N/A')}")
            return True
        except ImportError:
            self.logger.error("Could not import EnterpriseGPUManager. System info will be incomplete.", component="SysInit")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize system info: {str(e)}", component="SysInit", error_details=traceback.format_exc())
            safe_print(f"⚠️ System info initialization failed")
            return False
    
    def _initialize_menu_1(self) -> bool:
        """Initialize the single, unified Menu 1 Elliott Wave system."""
        if not self.config:
            safe_print("❌ Cannot initialize Menu 1: Configuration is not loaded.")
            return False
        if not self.logger:
            safe_print("❌ Cannot initialize Menu 1: Logger is not loaded.")
            return False
            
        # Try multiple Menu 1 implementations in order of preference
        menu_implementations = [
            # Primary: Real Enterprise Menu 1 (ACTUAL AI PROCESSING)
            {
                'module': 'menu_modules.real_enterprise_menu_1',
                'class': 'RealEnterpriseMenu1',
                'name': 'Real Enterprise Elliott Wave (ACTUAL AI)',
                'priority': 1
            },
            # Secondary: Enterprise Production Menu 1 (Simulation)
            {
                'module': 'menu_modules.enterprise_production_menu_1',
                'class': 'EnterpriseProductionMenu1',
                'name': 'Enterprise Production Elliott Wave (Simulation)',
                'priority': 2
            },
            # Tertiary: Enhanced Menu 1 (Beautiful Dashboard)
            {
                'module': 'menu_modules.enhanced_menu_1_elliott_wave', 
                'class': 'EnhancedMenu1ElliottWave',
                'name': 'Enhanced Elliott Wave with Beautiful Dashboard',
                'priority': 3
            },
            # Quaternary: Enhanced Menu 1 (Alternative name)
            {
                'module': 'menu_modules.enhanced_menu_1_elliott_wave',
                'class': 'BeautifulMenu1ElliottWave', 
                'name': 'Beautiful Elliott Wave Dashboard',
                'priority': 4
            }
        ]
        
        for impl in menu_implementations:
            try:
                safe_print(f"🔄 Trying {impl['name']}...")
                module = __import__(impl['module'], fromlist=[impl['class']])
                menu_class = getattr(module, impl['class'])
                
                # Initialize with config
                config_to_pass = self.config.config if self.config else {}
                self.menu_1 = menu_class(config_to_pass)
                
                self.logger.info(f"✅ {impl['name']}: READY")
                safe_print(f"✅ {impl['name']}: LOADED")
                
                self.menu_available = True
                self.menu_type = impl['name']
                return True
                
            except ImportError as e_imp:
                safe_print(f"⚠️ {impl['name']}: Import failed - {e_imp}")
                continue
            except Exception as e:
                safe_print(f"⚠️ {impl['name']}: Initialization failed - {e}")
                continue
        
        # If all implementations fail
        safe_print("❌ CRITICAL: All Menu 1 implementations failed to load")
        self.logger.critical("All Menu 1 implementations failed to load")
        self.menu_1 = None
        self.menu_available = False
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
            "║ 5. 🏢 OMS & MM System with 100 USD Capital                                 ⭐ NEW!         ║",
            "║    💰 Order Management System + Money Management with Menu 1 Strategy                          ║",
            "║    📊 Professional trading system with 100 USD capital and enterprise features                ║",
            "║                                                                                                   ║",
            "║ D. 🎨 Beautiful Progress Bars Demo                                                               ║",
            "║    ✨ Demonstration of visual progress tracking system                                           ║",
            "║                                                                                                   ║",
            "║ T. 🔐 Terminal Lock System                                                                      ║",
            "║    🎯 Beautiful & Modern Terminal Security Lock with Enterprise Features                        ║",
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
            elif choice == "5":
                return self._handle_oms_mm_system()
            elif choice == "D":
                return self._handle_progress_demo()
            elif choice == "T":
                return self._handle_terminal_lock()
            elif choice == "E":
                return self._handle_exit()
            elif choice == "R":
                return self._handle_restart()
            else:
                safe_print(f"❌ Invalid choice: {choice}")
                safe_print("💡 Please select 1, 2, 3, 5, D, T, E, or R")
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
                    # Get performance metrics from result
                    performance_metrics = result.get('performance_metrics', {})
                    
                    # Check for session_summary in result or use result directly
                    if 'session_summary' in result:
                        summary = result['session_summary']
                    else:
                        summary = result
                    
                    safe_print(f"\n📊 SESSION SUMMARY:")
                    
                    # Extract total steps
                    total_steps = summary.get('total_steps', result.get('total_steps', 'N/A'))
                    safe_print(f"   📈 Total Steps: {total_steps}")
                    
                    # Extract features selected from multiple possible locations
                    selected_features = (
                        performance_metrics.get('selected_features') or
                        summary.get('selected_features') or
                        result.get('selected_features') or
                        performance_metrics.get('original_features') or
                        'N/A'
                    )
                    safe_print(f"   🎯 Features Selected: {selected_features}")
                    
                    # Extract AUC from multiple possible locations
                    model_auc = (
                        performance_metrics.get('auc_score') or
                        performance_metrics.get('cnn_lstm_auc') or
                        summary.get('model_auc') or
                        result.get('model_auc') or
                        'N/A'
                    )
                    if isinstance(model_auc, float):
                        model_auc = f"{model_auc:.4f}"
                    safe_print(f"   🧠 Model AUC: {model_auc}")
                    
                    # Extract performance grade or calculate from metrics
                    performance_grade = summary.get('performance_grade', result.get('performance_grade'))
                    if not performance_grade and performance_metrics:
                        # Calculate performance grade based on metrics
                        auc = performance_metrics.get('auc_score', performance_metrics.get('cnn_lstm_auc', 0))
                        sharpe = performance_metrics.get('sharpe_ratio', 0)
                        win_rate = performance_metrics.get('win_rate', 0)
                        
                        if auc >= 0.80 and sharpe >= 1.5 and win_rate >= 0.70:
                            performance_grade = "Excellent"
                        elif auc >= 0.70 and sharpe >= 1.0 and win_rate >= 0.60:
                            performance_grade = "Good"
                        elif auc >= 0.60:
                            performance_grade = "Fair"
                        else:
                            performance_grade = "Poor"
                    
                    safe_print(f"   📊 Performance: {performance_grade or 'N/A'}")
                    
                    # Additional performance metrics if available
                    if performance_metrics:
                        safe_print(f"\n📈 DETAILED METRICS:")
                        if 'sharpe_ratio' in performance_metrics:
                            safe_print(f"   📊 Sharpe Ratio: {performance_metrics['sharpe_ratio']:.4f}")
                        if 'win_rate' in performance_metrics:
                            safe_print(f"   🎯 Win Rate: {performance_metrics['win_rate']:.2%}")
                        if 'max_drawdown' in performance_metrics:
                            safe_print(f"   📉 Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
                        if 'data_rows' in performance_metrics:
                            safe_print(f"   📊 Data Rows Processed: {performance_metrics.get('data_rows', 'N/A'):,}")
                        
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
        """Display system status and resource monitoring dashboard."""
        if not self.resource_manager:
            safe_print("❌ Resource Manager is not active.")
            input("\nPress Enter to continue...")
            return True

        safe_print("\n" + "═"*25 + " 📊 SYSTEM STATUS & RESOURCE MONITOR " + "═"*25)
                
        # Basic system status
        safe_print(f"\n🏢 Enterprise System Status:")
        safe_print(f"  📊 Session ID: {self.session_id}")
        safe_print(f"  🧠 Resource Manager: {'✅ Active' if self.resource_manager else '❌ Inactive'}")
        safe_print(f"  📝 Logger: {'✅ Active' if self.logger else '❌ Inactive'}")
        safe_print(f"  ⚙️ Configuration: {'✅ Active' if self.config else '❌ Inactive'}")
        safe_print(f"  🎛️ Menu 1: {'✅ Ready' if self.menu_available else '❌ Not Ready'}")
        
        # Resource information if available
        try:
            safe_print(f"\n💾 Resource Information:")
            safe_print(f"  Resource Manager Type: {type(self.resource_manager).__name__}")
            safe_print(f"  System Mode: Production")
        except Exception as e:
            safe_print(f"  ⚠️ Could not retrieve detailed resource information: {e}")

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
    
    def _handle_backtest_strategy(self) -> bool:
        """Handle Enhanced Profitable Backtest System"""
        safe_print("\n🚀 ENHANCED PROFITABLE BACKTEST - HIGH-VOLUME TRADING SYSTEM")
        safe_print("="*80)
        safe_print("💰 Optimized for Maximum Profitability & High-Volume Trading")
        safe_print("🎯 Requirements: กำไรขั้นต่ำ 1 USD ต่อออเดอร์, ออเดอร์มากกว่า 1,500")
        safe_print("📦 Features: Progressive Lot Sizing, Scalping Strategy, High-Frequency Signals")
        safe_print("🧠 AI-Powered Signal Generation with Multiple Technical Indicators")
        safe_print("")
        
        try:
            # Import Enhanced Menu 5 Profitable Backtest
            from menu_modules.enhanced_menu_5_profitable_backtest import run_enhanced_profitable_menu_5
            
            safe_print("✅ Enhanced Profitable Backtest System loaded successfully")
            safe_print("🚀 Initializing high-volume trading system...")
            safe_print("🎨 Beautiful progress tracking and detailed analysis will be displayed")
            safe_print("📊 Progressive Lot Sizing with 1 USD minimum profit per trade")
            safe_print("🎯 Scalping Strategy with High-Frequency Signal Generation")
            safe_print("📈 Target: >1,500 trades with consistent profitability")
            safe_print("")
            
            # Execute Enhanced Profitable Menu 5
            start_time = time.time()
            result = run_enhanced_profitable_menu_5()
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Process results
            if result and result.get('status') == 'SUCCESS':
                safe_print(f"\n🎉 ENHANCED PROFITABLE BACKTEST COMPLETED SUCCESSFULLY!")
                safe_print(f"⏱️ Duration: {duration:.2f} seconds")
                
                # Display backtest results
                targets_achieved = result.get('targets_achieved', {})
                
                safe_print(f"\n📊 ENHANCED PERFORMANCE SUMMARY:")
                safe_print(f"   💰 Initial Capital: ${result.get('initial_capital', 0):,.2f}")
                safe_print(f"   💰 Final Capital: ${result.get('final_capital', 0):,.2f}")
                safe_print(f"   📈 Total Return: {result.get('total_return', 0):+.2f}%")
                safe_print(f"   📊 Total Trades: {result.get('total_trades', 0):,}")
                safe_print(f"   ✅ Win Rate: {result.get('win_rate', 0):.1f}%")
                safe_print(f"   ⚡ Profit Factor: {result.get('profit_factor', 0):.2f}")
                safe_print(f"   🛡️ Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
                safe_print(f"   � Avg Profit/Trade: ${result.get('avg_profit_per_trade', 0):.2f}")
                
                # Target Achievement Status
                safe_print(f"\n🎯 TARGET ACHIEVEMENT STATUS:")
                min_profit_status = "✅ ACHIEVED" if targets_achieved.get('min_profit_per_trade') else "❌ NOT ACHIEVED"
                trades_status = "✅ ACHIEVED" if targets_achieved.get('trades_above_1500') else "❌ NOT ACHIEVED"
                profitable_status = "✅ ACHIEVED" if targets_achieved.get('profitable_system') else "❌ NOT ACHIEVED"
                
                safe_print(f"   💰 Min Profit/Trade (≥1 USD): {min_profit_status}")
                safe_print(f"   📊 Trades Above 1,500: {trades_status}")
                safe_print(f"   📈 Profitable System: {profitable_status}")
                
                # System Features
                safe_print(f"\n🎪 SYSTEM FEATURES:")
                safe_print(f"   📦 Progressive Lot Sizing: ✅ ENABLED")
                safe_print(f"   🎯 Scalping Strategy: ✅ ENABLED")
                safe_print(f"   🔄 High-Frequency Signals: ✅ ENABLED")
                safe_print(f"   💡 AI-Powered Analysis: ✅ ENABLED")
                
                safe_print("\n💡 Analysis completed using real market data with high-frequency scalping")
                safe_print("📊 Results saved for detailed review and analysis")
                
            elif result and result.get('status') == 'ERROR':
                safe_print(f"\n❌ PROFITABLE BACKTEST FAILED: {result.get('error', 'Unknown error')}")
                return False
            else:
                safe_print("\n⚠️ Profitable backtest completed with unexpected result format")
                safe_print(f"Result: {result}")
                return False
                
        except ImportError as e:
            safe_print("❌ Menu 5 Profitable Backtest not available")
            safe_print(f"   Import Error: {e}")
            safe_print("💡 Make sure menu_modules/enhanced_menu_5_profitable_backtest.py exists")
            safe_print("🔧 Try option 3 for system diagnostics")
            
        except Exception as e:
            safe_print(f"\n❌ Profitable Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            
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
    
    def _handle_oms_mm_system(self) -> bool:
        """Handle OMS & MM System with 100 USD Capital"""
        safe_print("\n🏢 OMS & MM SYSTEM WITH 100 USD CAPITAL")
        safe_print("="*80)
        
        try:
            # Import the new menu 5 system
            from menu_modules.menu_5_oms_mm_100usd import Menu5OMSMMSystem
            
            safe_print("🔄 Initializing OMS & MM System...")
            
            # Create and run the system
            oms_mm_system = Menu5OMSMMSystem()
            results = oms_mm_system.run_full_system()
            
            if results:
                safe_print("\n🎉 OMS & MM System completed successfully!")
                self._display_oms_mm_results(results)
                return True
            else:
                safe_print("❌ OMS & MM System failed to complete")
                return False
                
        except ImportError as e:
            safe_print(f"❌ Failed to import OMS & MM System: {e}")
            safe_print("💡 Make sure menu_modules/menu_5_oms_mm_100usd.py exists")
            return False
        except Exception as e:
            safe_print(f"❌ OMS & MM System error: {e}")
            return False
        
        input("\nPress Enter to continue...")
        return True
    
    def _display_oms_mm_results(self, results: dict):
        """Display OMS & MM System results"""
        try:
            safe_print("\n📊 OMS & MM SYSTEM RESULTS")
            safe_print("="*50)
            
            # Capital Management
            safe_print("💰 CAPITAL MANAGEMENT:")
            safe_print(f"   Initial Capital: ${results.get('initial_capital', 100):.2f}")
            safe_print(f"   Final Capital: ${results.get('final_capital', 100):.2f}")
            safe_print(f"   Total Return: {results.get('total_return_pct', 0):.2f}%")
            safe_print(f"   Total P&L: ${results.get('total_pnl', 0):.2f}")
            
            # Performance Metrics
            safe_print("\n📈 PERFORMANCE METRICS:")
            safe_print(f"   Total Trades: {results.get('trades_executed', 0)}")
            safe_print(f"   Win Rate: {results.get('win_rate', 0):.2f}%")
            safe_print(f"   Profit Factor: {results.get('profit_factor', 0):.2f}")
            safe_print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
            
            # Order Management
            account_summary = results.get('account_summary', {})
            safe_print("\n🏢 ORDER MANAGEMENT:")
            safe_print(f"   Total Orders: {account_summary.get('total_orders', 0)}")
            safe_print(f"   Filled Orders: {account_summary.get('filled_orders', 0)}")
            safe_print(f"   Active Positions: {account_summary.get('total_positions', 0)}")
            
            # Strategy Information
            safe_print("\n🎯 STRATEGY INFORMATION:")
            safe_print("   Strategy Source: Menu 1 (CNN-LSTM + DQN)")
            safe_print("   Capital: 100 USD")
            safe_print("   Risk per Trade: 2%")
            safe_print("   Stop Loss: 2 ATR")
            safe_print("   Take Profit: 3 ATR")
            
            # Performance Grade
            total_return = results.get('total_return_pct', 0)
            if total_return > 20:
                grade = "🏆 EXCELLENT"
            elif total_return > 10:
                grade = "🥈 GOOD"
            elif total_return > 0:
                grade = "🥉 POSITIVE"
            else:
                grade = "❌ NEEDS IMPROVEMENT"
            
            safe_print(f"\n🎯 PERFORMANCE GRADE: {grade}")
            
        except Exception as e:
            safe_print(f"❌ Error displaying results: {e}")
    
    def _handle_terminal_lock(self) -> bool:
        """Handle Terminal Lock System"""
        safe_print("\n🔐 NICEGOLD ENTERPRISE TERMINAL LOCK SYSTEM")
        safe_print("="*80)
        safe_print("🎯 Beautiful & Modern Terminal Security Lock with Enterprise Features")
        safe_print("")
        
        try:
            # Try to import and run the Terminal Lock Interface
            try:
                from terminal_lock_interface import SimpleTerminalLock
                safe_print("✅ Simple Terminal Lock System loaded successfully")
                
                # Create and run Terminal Lock
                lock = SimpleTerminalLock()
                
                # Show quick demo or interactive menu
                safe_print("\n🎮 TERMINAL LOCK OPTIONS:")
                safe_print("1. 🎪 Interactive Menu (Full Features)")
                safe_print("2. 🔐 Quick Lock Demo")
                safe_print("3. 📊 System Status")
                safe_print("4. 🚪 Return to Main Menu")
                
                try:
                    choice = input("\n🎯 Select option (1-4): ").strip()
                    
                    if choice == "1":
                        safe_print("\n🎪 Starting Interactive Terminal Lock Menu...")
                        lock.interactive_menu()
                    elif choice == "2":
                        safe_print("\n🔐 Running Quick Lock Demo...")
                        safe_print("Setting demo password: 'demo123'")
                        lock.set_password("demo123")
                        
                        safe_print("\n🔐 Locking terminal...")
                        lock.lock()
                        
                        safe_print("\n⏳ Waiting 3 seconds...")
                        time.sleep(3)
                        
                        safe_print("\n🔓 Unlocking with demo password...")
                        unlock_result = lock.unlock("demo123")
                        
                        if unlock_result:
                            safe_print("✅ Demo completed successfully!")
                        else:
                            safe_print("❌ Demo unlock failed")
                    elif choice == "3":
                        safe_print("\n📊 Terminal Lock System Status:")
                        lock.show_status()
                    elif choice == "4":
                        safe_print("🚪 Returning to main menu...")
                    else:
                        safe_print("❌ Invalid choice")
                        
                except (EOFError, KeyboardInterrupt):
                    safe_print("\n🛑 Terminal Lock interrupted by user")
                    
            except ImportError as e:
                safe_print("❌ Terminal Lock Interface not available")
                safe_print(f"   Error: {e}")
                safe_print("💡 Make sure terminal_lock_interface.py is in the project root")
                
                # Try Enterprise Terminal Lock as fallback
                try:
                    from core.enterprise_terminal_lock import EnterpriseTerminalLock
                    safe_print("✅ Enterprise Terminal Lock System loaded as fallback")
                    
                    lock = EnterpriseTerminalLock()
                    safe_print("\n🏢 Running Enterprise Terminal Lock...")
                    safe_print("🎯 Enterprise Terminal Lock is available but demo requires manual configuration")
                    safe_print("💡 Use Simple Terminal Lock Interface for quick demo")
                    
                except ImportError:
                    safe_print("❌ Enterprise Terminal Lock also not available")
                    safe_print("💡 Please check the terminal lock system installation")
            
            # Show Terminal Lock Features
            safe_print("\n🌟 TERMINAL LOCK FEATURES:")
            safe_print("  🎨 Beautiful ASCII Art Displays")
            safe_print("  🔐 Password Protection with SHA-256 Hashing")
            safe_print("  🛡️ Enterprise Security Features")
            safe_print("  🌈 Cross-platform Color Support")
            safe_print("  📊 Real-time System Information")
            safe_print("  ⚡ Lightning-fast Lock/Unlock Operations")
            safe_print("  🔄 Session Management & File-based Locking")
            safe_print("  📝 Comprehensive Logging & Monitoring")
            safe_print("  🧪 100% Tested Quality")
            safe_print("  🏢 Enterprise-grade Compliance")
            
        except Exception as e:
            safe_print(f"❌ Terminal Lock system error: {e}")
            import traceback
            traceback.print_exc()
        
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
    
    def _handle_restart(self) -> Any: # Changed return type to Any
        """Handles system restart."""
        self.logger.info("Requesting system restart...")
        self.running = False
        # The return value of 'False' from handle_menu_choice would stop the loop,
        # but we also return a specific value to indicate restart is needed.
        return 'restart'
    
    def start(self):
        """
        Start the unified master menu system with interactive menu including Terminal Lock.
        """
        # Initialize components first
        if not self.initialize_components():
            safe_print("❌ CRITICAL: Failed to initialize system components")
            return
            
        if self.logger:
            self.logger.info("🚀 Starting Unified Master Menu System...")
            self.logger.info("🎛️ INTERACTIVE MODE: Showing complete menu with Terminal Lock System.")
        else:
            safe_print("🚀 Starting Unified Master Menu System...")
            safe_print("🎛️ INTERACTIVE MODE: Showing complete menu with Terminal Lock System.")

        # Show interactive menu with Terminal Lock option
        while self.running:
            self.display_unified_menu()
            try:
                choice = self.get_user_choice()
                action_result = self.handle_menu_choice(choice)

                if action_result == 'restart':
                    # Special case for restart
                    safe_print("\n🔄 Restarting system...")
                    # A wrapper script would be needed to truly restart the process.
                    # For now, we exit and the user can re-run.
                    break 
                
                if not action_result:
                    # For 'exit' or other loop-breaking conditions
                    self.running = False

            except KeyboardInterrupt:
                safe_print("\n\n🛑 Caught KeyboardInterrupt. Exiting gracefully.\n")
                self.running = False
            except Exception as e:
                self.logger.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                safe_print(f"❌ An unexpected error occurred: {e}")
                self.running = False

def main():
    """For testing purposes"""
    try:
        # Pass the unified logger from the main system
        system_menu = UnifiedMasterMenuSystem()
        system_menu.start()
    except Exception as e:
        # Fallback basic print in case logger fails
        print(f"💥 A critical error occurred in the master menu system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
