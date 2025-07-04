#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - COMPREHENSIVE OPTIMIZED VERSION
Complete system with Menu 1 Elliott Wave Pipeline - 100% functional
Enterprise-grade AI Trading System with Real Data Processing

Features:
- Elliott Wave CNN-LSTM Pattern Recognition
- DQN Reinforcement Learning Agent
- SHAP + Optuna Feature Selection
- Enterprise ML Protection System
- Advanced Resource Management
- Real-time Progress Tracking
- Zero Error Policy
"""

# Import aggressive CUDA suppression first
from aggressive_cuda_suppression import suppress_all_output

print("🏢 NICEGOLD ENTERPRISE PROJECTP - COMPREHENSIVE SYSTEM")
print("=" * 80)

def main():
    """Comprehensive optimized main entry point with complete Menu 1 implementation"""
    
    # Import necessary modules without suppressing output during development
    import os
    import sys
    import gc
    import psutil
    from datetime import datetime
    import logging
    import traceback
    
    # Initialize safe logging first
    from safe_logger import initialize_safe_system_logging
    safe_logger = initialize_safe_system_logging()
    
    # Force minimal memory usage
    gc.set_threshold(100, 5, 5)
    
    print("🧠 Initializing Enterprise Systems...")
    print("=" * 60)
    
    # Initialize Enhanced 80% Resource Manager (Enterprise Strategy)
    resource_manager = None
    try:
        from core.enhanced_80_percent_resource_manager import Enhanced80PercentResourceManager
        resource_manager = Enhanced80PercentResourceManager(
            target_allocation=0.80  # 80% resource utilization target
        )
        print("✅ Enhanced 80% Resource Manager (80% RAM): ACTIVE")
    except Exception as e:
        print(f"⚠️ Enhanced 80% RM unavailable: {e}")
        try:
            from core.lightweight_resource_manager import LightweightResourceManager
            resource_manager = LightweightResourceManager(memory_percentage=0.8, cpu_percentage=0.35)
            print("✅ Lightweight Resource Manager (80% RAM): ACTIVE")
        except Exception as e2:
            print(f"⚠️ Lightweight RM unavailable: {e2}")
            try:
                from core.high_memory_resource_manager import HighMemoryResourceManager
                resource_manager = HighMemoryResourceManager(memory_percentage=0.8, cpu_percentage=0.35)
                print("✅ High Memory Resource Manager (80% RAM): ACTIVE")
            except Exception as e3:
                print(f"⚠️ All resource managers unavailable: {e3}")
                resource_manager = None
    
    # Initialize Enterprise Logging System
    logger = None
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        logger = get_terminal_logger()
        logger.success("✅ Advanced Enterprise Logging active", "Startup")
        print("✅ Enterprise Logging: ACTIVE")
    except Exception as e:
        print(f"⚠️ Advanced logging unavailable: {e}")
        # Create safe console logger
        import logging
        logger = logging.getLogger("NICEGOLD_ENTERPRISE")
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.info("✅ Basic Enterprise Logging initialized")
        print("✅ Basic Logging: ACTIVE")
    
    # Enterprise Configuration with 80% Resource Strategy
    config = {
        'enterprise_mode': True,
        'resource_manager': resource_manager,
        'target_memory_usage': 0.8,      # 80% RAM target
        'target_cpu_usage': 0.35,        # 35% CPU for enhanced performance
        'zero_errors_mode': True,
        'real_data_only': True,
        'enterprise_ml_protection': True,
        'elliott_wave_cnn_lstm': True,
        'dqn_reinforcement_learning': True,
        'shap_optuna_features': True,
        'auc_target': 0.75,              # Upgraded to 75% AUC target
        'enhanced_performance': True,
        'resource_utilization_strategy': '80_percent'
    }
    
    print("🎛️ Loading Enterprise Elliott Wave Menu System...")
    print("=" * 60)
    
    # Initialize Enterprise Elliott Wave Menu 1 with Multiple Fallbacks
    menu_1 = None
    menu_available = False
    menu_type = "None"
    
    # Priority 1: Try Elliott Wave Menu 1 (Primary - uses real CSV data)
    try:
        with suppress_all_output():
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        
        menu_1 = Menu1ElliottWave(config, logger, resource_manager)
        print("✅ Elliott Wave Menu 1 (Enterprise): READY")
        menu_available = True
        menu_type = "Elliott Wave Enterprise"
        
    except Exception as e:
        print(f"⚠️ Elliott Wave menu failed: {e}")
        
        # Priority 2: Try Enhanced 80% Menu
        try:
            with suppress_all_output():
                from menu_modules.enhanced_80_percent_menu_1 import Enhanced80PercentMenu1
            
            menu_1 = Enhanced80PercentMenu1(config, logger, resource_manager)
            print("✅ Enhanced 80% Menu 1: READY")
            menu_available = True
            menu_type = "Enhanced 80% Resource"
            
        except Exception as e2:
            print(f"⚠️ Enhanced 80% menu failed: {e2}")
            
            # Priority 3: Try High Memory Menu
            try:
                with suppress_all_output():
                    from menu_modules.high_memory_menu_1 import HighMemoryMenu1
                
                menu_1 = HighMemoryMenu1(config, logger, resource_manager)
                print("✅ High Memory Menu 1: READY")
                menu_available = True
                menu_type = "High Memory"
                
            except Exception as e3:
                print(f"⚠️ High Memory menu failed: {e3}")
                
                # Priority 4: Try Optimized Menu (Final fallback)
                try:
                    with suppress_all_output():
                        from menu_modules.optimized_menu_1_elliott_wave import OptimizedMenu1ElliottWave
                    
                    menu_1 = OptimizedMenu1ElliottWave(config, logger, resource_manager)
                    print("✅ Optimized Menu 1: READY (fallback)")
                    menu_available = True
                    menu_type = "Optimized Fallback"
                    
                except Exception as e4:
                    print(f"❌ All menu systems failed: {e4}")
                    menu_available = False
    
    # Check if any menu is available
    if not menu_available:
        print("🚨 CRITICAL ERROR: No menu system available")
        print("🔧 Please check system installation:")
        print("   1. Verify menu_modules/ directory exists")
        print("   2. Check Elliott Wave modules in elliott_wave_modules/") 
        print("   3. Ensure core/ modules are properly installed")
        print("   4. Validate datacsv/ contains real market data")
        return
    
    # Display Enterprise System Status
    print("\n" + "="*80)
    print("🏢 NICEGOLD ENTERPRISE - ELLIOTT WAVE AI TRADING SYSTEM")
    print("="*80)
    print(f"🎛️ Menu System: {menu_type}")
    print(f"🧠 Resource Manager: {resource_manager.__class__.__name__ if resource_manager else 'Basic'}")
    print(f"📝 Logging: {'Enterprise Advanced' if 'advanced' in str(type(logger)) else 'Enterprise Basic'}")
    print(f"🎯 Target Resource Usage: 80% RAM, 35% CPU")
    print(f"📊 Data Source: Real Market Data (XAUUSD)")
    print(f"🤖 AI Components: CNN-LSTM + DQN + SHAP/Optuna")
    print("="*80)
    print("1. 🌊 Elliott Wave Full Pipeline (Enhanced 80% Utilization)")
    print("2. 📊 System Status & Resource Monitor")
    print("0. 🚪 Exit")
    print("="*80)
    
    # Check if we're in an interactive environment
    def safe_input(prompt=""):
        """Safe input function that works in all environments"""
        import sys
        import select
        
        try:
            # Check if stdin is available and not redirected
            if not sys.stdin.isatty():
                print("⚠️ Non-interactive environment detected. Running Menu 1 automatically.")
                return "1"
            
            # Check if input is available (for Unix-like systems)
            if hasattr(select, 'select'):
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    print(prompt, end='', flush=True)
            else:
                print(prompt, end='', flush=True)
            
            # Try to read input with timeout
            try:
                line = sys.stdin.readline()
                if not line:  # EOF
                    print("\n⚠️ EOF detected. Running Menu 1 automatically.")
                    return "1"
                return line.strip()
            except (EOFError, OSError):
                print("\n⚠️ Input error detected. Running Menu 1 automatically.")
                return "1"
                
        except Exception as e:
            print(f"\n⚠️ Input system error: {e}. Running Menu 1 automatically.")
            return "1"
    
    # Auto-run counter for non-interactive environments
    import sys
    auto_run_count = 0
    max_auto_runs = 1
    
    while True:
        try:
            # For non-interactive environments, auto-run once then exit
            if not sys.stdin.isatty() and auto_run_count >= max_auto_runs:
                print("✅ Auto-run completed. Exiting.")
                break
            
            choice = safe_input("\n🎯 Select option (0-2): ").strip()
            
            # Handle auto-run for non-interactive environments
            if not sys.stdin.isatty():
                auto_run_count += 1
            
            if choice == "1":
                print("\n🚀 Starting Enhanced 80% Elliott Wave Pipeline...")
                try:
                    start_time = datetime.now()
                    
                    # Set resource manager to 80% if available
                    if resource_manager and hasattr(resource_manager, 'start_monitoring'):
                        resource_manager.start_monitoring()
                        print("📊 80% Resource monitoring activated")
                    
                    result = menu_1.run()
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    if result.get('success'):
                        print(f"✅ Enhanced 80% Pipeline completed successfully in {duration:.2f}s")
                        if 'performance' in result:
                            perf = result['performance']
                            print(f"📊 Performance Metrics: {perf}")
                        if 'resource_usage' in result:
                            usage = result['resource_usage']
                            print(f"🧠 Resource Usage: {usage}")
                    else:
                        print(f"❌ Pipeline failed: {result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"❌ Pipeline execution error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Only wait for input in interactive environments
                if sys.stdin.isatty():
                    safe_input("\nPress Enter to continue...")
                else:
                    print("\n⚠️ Non-interactive environment - continuing automatically...")
            
            elif choice == "2":
                print("\n📊 ENHANCED SYSTEM STATUS & RESOURCE MONITOR")
                print("=" * 60)
                
                # System Resource Status
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                
                print(f"💾 Memory Status:")
                print(f"   Current Usage: {memory.percent:.1f}% ({memory.used/(1024**3):.1f}GB/{memory.total/(1024**3):.1f}GB)")
                print(f"   Available: {memory.available/(1024**3):.1f}GB")
                print(f"   Target (80%): {memory.total * 0.8 / (1024**3):.1f}GB")
                
                if memory.percent >= 80:
                    print("   Status: ✅ 80%+ utilization achieved")
                elif memory.percent >= 70:
                    print("   Status: ⚠️ Good utilization (70%+)")
                else:
                    print("   Status: ❌ Under-utilized (<70%)")
                
                print(f"\n🖥️ CPU Status:")
                print(f"   Current Usage: {cpu:.1f}%")
                print(f"   Target: 35%")
                
                # Enhanced Resource Manager Status
                if resource_manager:
                    print(f"\n🧠 Enhanced 80% Resource Manager:")
                    try:
                        if hasattr(resource_manager, 'get_health_status'):
                            health = resource_manager.get_health_status()
                            print(f"   Health Score: {health.get('health_score', 0)}%")
                            print(f"   Status: {'✅ Excellent' if health.get('health_score', 0) >= 90 else '⚠️ Good' if health.get('health_score', 0) >= 70 else '❌ Needs attention'}")
                            
                            if 'current_allocation' in health:
                                alloc = health['current_allocation'] * 100
                                print(f"   Current Allocation: {alloc:.1f}%")
                                
                            if 'target_allocation' in health:
                                target = health['target_allocation'] * 100
                                print(f"   Target Allocation: {target:.1f}%")
                                
                            if 'memory_efficiency' in health:
                                print(f"   Memory Efficiency: {health['memory_efficiency']:.1f}%")
                                
                            if 'performance_score' in health:
                                print(f"   Performance Score: {health['performance_score']:.1f}%")
                                
                        elif hasattr(resource_manager, 'get_resource_status'):
                            status = resource_manager.get_resource_status()
                            print(f"   Status: {status}")
                        else:
                            print("   Status: ✅ Active (basic)")
                            
                        # Check if 80% strategy is working
                        if hasattr(resource_manager, 'memory_percentage'):
                            print(f"   Memory Target: {resource_manager.memory_percentage * 100}%")
                        if hasattr(resource_manager, 'cpu_percentage'):
                            print(f"   CPU Target: {resource_manager.cpu_percentage * 100}%")
                            
                    except Exception as e:
                        print(f"   Status: ⚠️ Active (status error: {e})")
                else:
                    print(f"\n🧠 Resource Manager: ❌ Not available")
                
                # Menu System Status
                print(f"\n🎛️ Menu System Status:")
                print(f"   Type: {menu_type}")
                print(f"   Status: {'✅ Available' if menu_available else '❌ Unavailable'}")
                
                # Configuration Status
                print(f"\n⚙️ Configuration Status:")
                print(f"   Memory Target: {config.get('target_memory_usage', 0) * 100}%")
                print(f"   CPU Target: {config.get('target_cpu_usage', 0) * 100}%")
                print(f"   AUC Target: {config.get('auc_target', 0) * 100}%")
                print(f"   Strategy: {config.get('resource_utilization_strategy', 'standard')}")
                
                # Performance Recommendations
                print(f"\n🎯 Performance Analysis:")
                if memory.percent < 70:
                    print("   📈 Recommendation: System can utilize more RAM for better performance")
                    print("   💡 Suggestion: Increase data batch sizes or model complexity")
                elif memory.percent >= 80:
                    print("   ✅ Excellent: Optimal 80% RAM utilization achieved")
                else:
                    print("   ✅ Good: RAM utilization is within acceptable range")
                
                if cpu < 30:
                    print("   📈 CPU has capacity for more intensive processing")
                elif cpu >= 35:
                    print("   ✅ CPU utilization is at target level")
                
                # Only wait for input in interactive environments
                if sys.stdin.isatty():
                    safe_input("\nPress Enter to continue...")
                else:
                    print("\n⚠️ Non-interactive environment - continuing automatically...")
            
            elif choice == "0":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 0-2.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Menu error: {e}")
    
    # Cleanup
    if resource_manager:
        try:
            if hasattr(resource_manager, 'stop_monitoring'):
                resource_manager.stop_monitoring()
            if hasattr(resource_manager, 'cleanup'):
                resource_manager.cleanup()
        except Exception as e:
            print(f"⚠️ Resource manager cleanup warning: {e}")
    
    # Final garbage collection
    gc.collect()
    print("✅ NICEGOLD Enterprise Enhanced 80% System Shutdown Complete")

if __name__ == "__main__":
    main()
