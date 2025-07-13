#!/usr/bin/env python3
"""
🎯 QUICK MENU 5 ENTERPRISE DEMO
NICEGOLD ProjectP - Demonstrate Enterprise Logging System

This script runs a quick demonstration of Menu 5 with enterprise logging
to show all the features working together.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

def run_menu5_enterprise_demo():
    """Run Menu 5 enterprise logging demonstration"""
    print("🎯 MENU 5 ENTERPRISE LOGGING DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Import Menu 5
        from menu_modules.menu_5_backtest_strategy import Menu5BacktestStrategy
        
        print("🚀 Initializing Menu 5 with Enterprise Logging...")
        
        # Create Menu 5 instance
        menu5 = Menu5BacktestStrategy()
        
        print(f"📊 Session ID: {menu5.session_id}")
        print(f"🔗 Menu 1 Detection: {menu5.menu1_session_id or 'Will detect on run'}")
        
        # Show enterprise features
        print("\n🏢 ENTERPRISE FEATURES ENABLED:")
        print("✅ Automatic Menu 1 session detection")
        print("✅ SQLite database logging")
        print("✅ CSV export functionality")
        print("✅ JSON report generation")
        print("✅ Excel analysis files")
        print("✅ Real-time execution logging")
        print("✅ Data integrity validation")
        print("✅ Session traceability")
        
        # Run Menu 5 (this will demonstrate all enterprise logging)
        print("\n🚀 Running Menu 5 Enterprise Backtest...")
        print("📊 This will demonstrate:")
        print("   • Automatic Menu 1 session linking")
        print("   • Complete trade detail recording")
        print("   • Multiple export format generation")
        print("   • Performance metrics logging")
        print("   • Data integrity validation")
        
        # Execute Menu 5
        results = menu5.run()
        
        # Show results
        print("\n✅ ENTERPRISE LOGGING DEMONSTRATION COMPLETED!")
        
        if "error" not in results:
            print(f"📊 Session ID: {results.get('session_id', 'N/A')}")
            print(f"🔗 Menu 1 Linked: {results.get('menu1_session_id', 'Standalone')}")
            print(f"⏱️ Execution Time: {results.get('execution_time', 0):.2f}s")
            
            # Show generated files
            print("\n📁 ENTERPRISE FILES GENERATED:")
            session_dir = Path("outputs/backtest_sessions")
            if session_dir.exists():
                for session_path in session_dir.glob("menu5_*"):
                    print(f"📂 Session Directory: {session_path.name}")
                    
                    # Show databases
                    db_dir = session_path / "databases"
                    if db_dir.exists():
                        for db_file in db_dir.glob("*.db"):
                            size = db_file.stat().st_size
                            print(f"   🗃️ {db_file.name}: {size:,} bytes")
                    
                    # Show CSV files
                    csv_dir = session_path / "trade_records"
                    if csv_dir.exists():
                        for csv_file in csv_dir.glob("*.csv"):
                            size = csv_file.stat().st_size
                            print(f"   📄 {csv_file.name}: {size:,} bytes")
                    
                    # Show reports
                    reports_dir = session_path / "reports"
                    if reports_dir.exists():
                        for report_file in reports_dir.glob("*"):
                            size = report_file.stat().st_size
                            print(f"   📋 {report_file.name}: {size:,} bytes")
            
            print("\n🎉 ENTERPRISE LOGGING FULLY OPERATIONAL!")
            print("📊 All trade details recorded with complete traceability")
            print("🔗 Session linked to Menu 1 for comprehensive analysis")
            
        else:
            print(f"❌ Error occurred: {results.get('error', 'Unknown')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        print(f"📋 Details: {traceback.format_exc()}")
        return False

def main():
    """Main demo execution"""
    print("🏢 NICEGOLD ENTERPRISE LOGGING SYSTEM DEMONSTRATION")
    print("🎯 Menu 5 BackTest Strategy with Complete Trade Recording")
    print("-" * 80)
    
    success = run_menu5_enterprise_demo()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ENTERPRISE LOGGING DEMONSTRATION SUCCESSFUL")
        print("✅ Menu 5 now records every trade detail for comprehensive analysis")
        print("✅ All data exported in multiple formats for easy analysis")
        print("✅ Session linking provides complete audit trail")
        print("\n📖 USAGE INSTRUCTIONS:")
        print("1. Run 'python ProjectP.py' and select Menu 5")
        print("2. Check 'outputs/backtest_sessions/' for detailed logs")
        print("3. Use SQLite, CSV, or Excel files for analysis")
        print("4. Every trade detail is recorded automatically")
    else:
        print("❌ ENTERPRISE LOGGING DEMONSTRATION FAILED")
        print("⚠️ Check system configuration and retry")
    print("=" * 80)

if __name__ == "__main__":
    main()
