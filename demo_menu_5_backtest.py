#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MENU 5 BACKTEST STRATEGY - DEMONSTRATION SCRIPT
Quick         safe_print("🎯 TRADING CONDITIONS THAT WILL BE SIMULATED:")
        safe_print("-" * 50)
        safe_print("💰 Spread: 100 points (realistic market spread)")
        safe_print("💵 Commission: $0.07 per 0.01 lot (professional rate)")
        safe_print("⚡ Slippage: 1-3 points (realistic execution delay)")stration of Menu 5 functionality

🎮 Features Demonstrated:
✅ Professional Trading Simulation
✅ 10 Sessions Analysis  
✅ Latest Session Detection
✅ Beautiful Progress Tracking
✅ Comprehensive Results Display
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def safe_print(*args, **kwargs):
    """Safe print with error handling"""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        try:
            message = " ".join(map(str, args))
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except:
            pass

def demo_menu_5():
    """Demonstrate Menu 5 BackTest Strategy"""
    safe_print("🎯 MENU 5 BACKTEST STRATEGY - LIVE DEMONSTRATION")
    safe_print("="*70)
    safe_print("💰 Professional Trading Simulation with Realistic Market Conditions")
    safe_print("📊 100-point spread | $0.07 commission per 0.01 lot")
    safe_print("📈 10 sessions analysis with latest session detection")
    safe_print("")
    
    try:
        # Import and initialize Menu 5
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        safe_print("✅ Menu 5 BackTest Strategy loaded successfully")
        safe_print("🚀 Initializing professional trading simulation...")
        
        # Initialize with demo config
        config = {
            'demo_mode': True,
            'session_limit': 5,  # Limit to 5 sessions for demo
            'fast_execution': True
        }
        
        menu_5 = Menu5BacktestStrategy(config)
        safe_print("🎮 Demo configuration applied")
        safe_print("")
        
        # Show what Menu 5 will analyze
        safe_print("🔍 DEMO PREVIEW - What Menu 5 Will Analyze:")
        safe_print("-" * 50)
        
        # Check available sessions
        sessions_dir = Path("outputs/sessions")
        if sessions_dir.exists():
            sessions = sorted(list(sessions_dir.glob("20*")), reverse=True)
            demo_sessions = sessions[:5]  # Show top 5 for demo
            
            safe_print(f"📊 Available Sessions: {len(sessions)} total")
            safe_print(f"🎯 Demo will analyze: {len(demo_sessions)} latest sessions")
            safe_print("")
            
            for i, session in enumerate(demo_sessions, 1):
                date_part = session.name[:8]
                time_part = session.name[9:]
                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                
                status = "📅 LATEST" if i == 1 else f"📈 Session {i}"
                safe_print(f"   {status}: {formatted_date} {formatted_time}")
                
                # Show session data if available
                session_summary = session / "session_summary.json"
                if session_summary.exists():
                    try:
                        import json
                        with open(session_summary, 'r') as f:
                            summary = json.load(f)
                        
                        auc = summary.get('model_auc', 'N/A')
                        steps = summary.get('total_steps', 'N/A')
                        grade = summary.get('performance_grade', 'N/A')
                        
                        safe_print(f"      🧠 AUC: {auc} | 📊 Steps: {steps} | 🏆 Grade: {grade}")
                    except:
                        safe_print(f"      📋 Session data available")
                else:
                    safe_print(f"      ⚠️ Session summary not found")
                    
        else:
            safe_print("⚠️ No sessions directory found")
            safe_print("💡 Run Menu 1 first to generate session data")
            
        safe_print("")
        safe_print("🎯 TRADING CONDITIONS THAT WILL BE SIMULATED:")
        safe_print("-" * 50)
        safe_print("💰 Spread: 100 points (realistic market spread)")
        safe_print("💵 Commission: $0.07 per 0.01 lot (professional rate)")
        safe_print("⚡ Slippage: 1-3 points (realistic execution delay)")
        safe_print("📈 Order Types: Market, Limit, Stop orders")
        safe_print("🎯 Position Sizing: Professional risk management")
        
        safe_print("")
        safe_print("📊 PERFORMANCE METRICS THAT WILL BE CALCULATED:")
        safe_print("-" * 50)
        safe_print("🔄 Total Trades Executed")
        safe_print("💚 Profitable vs Losing Trades")
        safe_print("🎯 Win Rate Percentage")
        safe_print("💎 Total Profit/Loss")
        safe_print("📊 Profit Factor")
        safe_print("📉 Maximum Drawdown")
        safe_print("⚡ Sharpe Ratio")
        safe_print("🏆 Risk-Adjusted Returns")
        
        safe_print("")
        safe_print("🎨 BEAUTIFUL FEATURES YOU'LL SEE:")
        safe_print("-" * 50)
        safe_print("🌈 Color-coded progress bars")
        safe_print("📊 Real-time performance dashboard")
        safe_print("⚡ Live statistics updates")
        safe_print("🎯 Professional results display")
        safe_print("📁 Comprehensive output files")
        safe_print("🏢 Enterprise-grade reporting")
        
        safe_print("")
        safe_print("🚀 READY TO RUN FULL MENU 5?")
        safe_print("="*70)
        safe_print("💡 To experience the full Menu 5 BackTest Strategy:")
        safe_print("1. Run: python ProjectP.py")
        safe_print("2. Select option '5' for BackTest Strategy")
        safe_print("3. Watch the beautiful progress bars and analytics!")
        safe_print("")
        safe_print("✨ This will give you the complete professional trading simulation")
        safe_print("📊 With all performance metrics and beautiful visualizations")
        
    except ImportError as e:
        safe_print("❌ Menu 5 not available for demo")
        safe_print(f"   Error: {e}")
        safe_print("💡 Make sure menu_modules/menu_5_backtest_strategy.py exists")
        
    except Exception as e:
        safe_print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

def show_menu_5_features():
    """Show comprehensive Menu 5 features"""
    safe_print("\n🌟 MENU 5 BACKTEST STRATEGY - COMPREHENSIVE FEATURES")
    safe_print("="*70)
    
    features = [
        ("🎯 Professional Trading Simulation", "Realistic market conditions with spread, commission, slippage"),
        ("📊 10 Sessions Analysis", "Automatic detection and analysis of latest trading sessions"),
        ("💰 Realistic Trading Costs", "100-point spread, $0.07 commission per 0.01 lot"),
        ("📅 Latest Session Detection", "Automatic identification of most recent session with timestamps"),
        ("🎨 Beautiful Progress Tracking", "Rich console UI with color-coded progress bars"),
        ("📈 Comprehensive Analytics", "Win rate, profit factor, Sharpe ratio, drawdown analysis"),
        ("🏢 Enterprise Integration", "Seamless integration with unified menu system"),
        ("📁 Professional Reporting", "JSON/CSV export with comprehensive trading reports"),
        ("⚡ Fast Performance", "Optimized execution with real-time monitoring"),
        ("🛡️ Error Handling", "Enterprise-grade error recovery and logging")
    ]
    
    for feature, description in features:
        safe_print(f"   {feature}")
        safe_print(f"      {description}")
        safe_print("")

def main():
    """Main demonstration function"""
    safe_print("🎮 MENU 5 BACKTEST STRATEGY - COMPLETE DEMONSTRATION")
    safe_print("="*70)
    safe_print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("🎯 Demonstrating professional trading simulation capabilities")
    safe_print("")
    
    # Show features overview
    show_menu_5_features()
    
    # Run main demonstration
    demo_menu_5()

if __name__ == "__main__":
    main()
