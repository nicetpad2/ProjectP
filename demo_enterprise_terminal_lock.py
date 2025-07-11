#!/usr/bin/env python3
"""
🎮 NICEGOLD Enterprise Terminal Lock - Demo System
Beautiful, Modern Terminal Locking Demonstration

Version: 1.0 Enterprise Edition
Date: 11 July 2025
Status: Production Ready
"""

import os
import sys
import time
import threading
from pathlib import Path

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent))

# Import ระบบ Terminal Lock
try:
    from core.enterprise_terminal_lock import EnterpriseTerminalLock
    LOCK_AVAILABLE = True
except ImportError:
    LOCK_AVAILABLE = False
    print("⚠️  Terminal Lock system not available. Please install required dependencies.")

# Rich library for beautiful output
RICH_AVAILABLE = False
Console = None
Panel = None
Table = None
Text = None
Align = None
Prompt = None
DOUBLE = ROUNDED = HEAVY = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.live import Live
    from rich.layout import Layout
    from rich.box import DOUBLE, ROUNDED, HEAVY
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.status import Status
    from rich.rule import Rule
    from rich.markdown import Markdown
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    print("⚠️  Rich library not available. Installing...")
    os.system("pip install rich")
    
    # Try importing again after installation
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        from rich.text import Text
        from rich.align import Align
        from rich.columns import Columns
        from rich.live import Live
        from rich.layout import Layout
        from rich.box import DOUBLE, ROUNDED, HEAVY
        from rich.prompt import Prompt, Confirm
        from rich.tree import Tree
        from rich.status import Status
        from rich.rule import Rule
        from rich.markdown import Markdown
        from rich import print as rprint
        RICH_AVAILABLE = True
    except ImportError:
        # Rich not available, use fallback
        pass


class TerminalLockDemo:
    """
    🎮 Demo System สำหรับ Enterprise Terminal Lock
    """
    
    def __init__(self):
        self.console = Console(force_terminal=True, width=120) if RICH_AVAILABLE else None
        self.lock_system = None
        
        # สี Theme
        self.theme = {
            'primary': 'bright_cyan',
            'secondary': 'bright_magenta',
            'success': 'bright_green',
            'warning': 'bright_yellow',
            'error': 'bright_red',
            'info': 'bright_blue'
        }
        
        # ASCII Art
        self.banner = """
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║  🎮 NICEGOLD ENTERPRISE TERMINAL LOCK SYSTEM - DEMO                                          ║
║                                                                                              ║
║  ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗                              ║
║  ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║                              ║
║     ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║                              ║
║     ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║                              ║
║     ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗                         ║
║     ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝                         ║
║                                                                                              ║
║  ██╗      ██████╗  ██████╗██╗  ██╗    ██████╗ ███████╗███╗   ███╗ ██████╗                  ║
║  ██║     ██╔═══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗                 ║
║  ██║     ██║   ██║██║     █████╔╝     ██║  ██║█████╗  ██╔████╔██║██║   ██║                 ║
║  ██║     ██║   ██║██║     ██╔═██╗     ██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║                 ║
║  ███████╗╚██████╔╝╚██████╗██║  ██╗    ██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝                 ║
║  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝                  ║
║                                                                                              ║
║                           🚀 Enterprise Production Ready                                     ║
║                              ✨ Beautiful & Modern                                          ║
║                               🔐 Security First                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
        """
    
    def show_banner(self):
        """แสดงป้ายแบนเนอร์"""
        if RICH_AVAILABLE:
            self.console.clear()
            banner_panel = Panel(
                Align.center(
                    Text(self.banner, style="bold bright_cyan"),
                    vertical="middle"
                ),
                box=HEAVY,
                style="bright_cyan"
            )
            self.console.print(banner_panel)
        else:
            print(self.banner)
        
        time.sleep(2)
    
    def show_features(self):
        """แสดงรายการฟีเจอร์"""
        if not RICH_AVAILABLE:
            print("🌟 Terminal Lock System Features:")
            print("• Beautiful modern interface")
            print("• Enterprise-grade security")
            print("• Real-time system monitoring")
            print("• Session management")
            print("• Password protection")
            print("• Beautiful animations")
            return
        
        features_table = Table(show_header=True, header_style="bold bright_white")
        features_table.add_column("Feature", style="bright_cyan")
        features_table.add_column("Description", style="bright_white")
        features_table.add_column("Status", style="bright_green")
        
        features = [
            ("🎨 Modern Interface", "Beautiful Rich-based terminal UI", "✅ Active"),
            ("🔐 Security System", "Enterprise-grade protection", "✅ Active"),
            ("📊 Real-time Monitoring", "Live system statistics", "✅ Active"),
            ("📋 Session Management", "Complete session tracking", "✅ Active"),
            ("🔑 Password Protection", "Secure unlock mechanism", "✅ Active"),
            ("🌈 Animations", "Beautiful lock/unlock animations", "✅ Active"),
            ("📱 Cross-platform", "Works on Windows/Linux/macOS", "✅ Active"),
            ("⚡ Performance", "Optimized for speed", "✅ Active"),
            ("🎯 Enterprise Ready", "Production-grade quality", "✅ Active"),
            ("🌟 User Experience", "Intuitive and beautiful", "✅ Active")
        ]
        
        for feature, description, status in features:
            features_table.add_row(feature, description, status)
        
        features_panel = Panel(
            features_table,
            title="🌟 Terminal Lock System Features",
            box=ROUNDED,
            style="bright_blue"
        )
        
        self.console.print(features_panel)
    
    def demo_lock_unlock(self):
        """สาธิตการล็อคและปลดล็อค"""
        if not LOCK_AVAILABLE:
            print("❌ Terminal Lock system not available!")
            return
        
        if RICH_AVAILABLE:
            self.console.print("\n🎮 [bold bright_yellow]Starting Lock/Unlock Demo...[/bold bright_yellow]")
        else:
            print("\n🎮 Starting Lock/Unlock Demo...")
        
        # สร้างระบบล็อค
        with EnterpriseTerminalLock() as lock_system:
            self.lock_system = lock_system
            
            # ตั้งรหัสผ่าน Demo
            lock_system.set_password("demo123")
            
            if RICH_AVAILABLE:
                self.console.print("🔑 [bright_green]Demo password set to: 'demo123'[/bright_green]")
            else:
                print("🔑 Demo password set to: 'demo123'")
            
            time.sleep(2)
            
            # Demo Lock
            if RICH_AVAILABLE:
                self.console.print("🔐 [bright_red]Locking terminal...[/bright_red]")
            else:
                print("🔐 Locking terminal...")
            
            lock_system.lock()
            
            # รอสักครู่เพื่อให้เห็นหน้าจอล็อค
            time.sleep(5)
            
            # Demo Unlock
            if RICH_AVAILABLE:
                self.console.print("🔓 [bright_green]Auto-unlocking with demo password...[/bright_green]")
            else:
                print("🔓 Auto-unlocking with demo password...")
            
            unlock_success = lock_system.unlock("demo123")
            
            if unlock_success:
                if RICH_AVAILABLE:
                    self.console.print("✅ [bold bright_green]Demo completed successfully![/bold bright_green]")
                else:
                    print("✅ Demo completed successfully!")
            else:
                if RICH_AVAILABLE:
                    self.console.print("❌ [bold bright_red]Demo failed![/bold bright_red]")
                else:
                    print("❌ Demo failed!")
    
    def demo_monitoring(self):
        """สาธิตการตรวจสอบระบบ"""
        if not RICH_AVAILABLE:
            print("📊 System Monitoring Demo (Basic)")
            return
        
        self.console.print("\n📊 [bold bright_cyan]System Monitoring Demo[/bold bright_cyan]")
        
        # สร้างตารางการตรวจสอบ
        monitoring_table = Table(show_header=True, header_style="bold bright_white")
        monitoring_table.add_column("Metric", style="bright_cyan")
        monitoring_table.add_column("Value", style="bright_white")
        monitoring_table.add_column("Trend", style="bright_green")
        
        # ข้อมูลจำลอง
        import psutil
        import random
        
        for i in range(5):
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            monitoring_table.add_row(
                f"CPU Usage #{i+1}",
                f"{cpu_usage:.1f}%",
                "📈" if random.random() > 0.5 else "📉"
            )
            
            monitoring_table.add_row(
                f"Memory Usage #{i+1}",
                f"{memory.percent:.1f}%",
                "📊" if memory.percent < 80 else "⚠️"
            )
            
            time.sleep(1)
        
        monitoring_panel = Panel(
            monitoring_table,
            title="📊 Real-time System Monitoring",
            box=ROUNDED,
            style="bright_magenta"
        )
        
        self.console.print(monitoring_panel)
    
    def demo_session_management(self):
        """สาธิตการจัดการ Session"""
        if not LOCK_AVAILABLE:
            print("📋 Session Management Demo not available!")
            return
        
        with EnterpriseTerminalLock() as lock_system:
            if RICH_AVAILABLE:
                self.console.print("\n📋 [bold bright_blue]Session Management Demo[/bold bright_blue]")
            else:
                print("\n📋 Session Management Demo")
            
            # สร้าง Session ทดสอบ
            lock_system.set_password("session123")
            
            # ล็อคและปลดล็อคหลายครั้ง
            for i in range(3):
                if RICH_AVAILABLE:
                    self.console.print(f"🔄 [bright_yellow]Creating session {i+1}...[/bright_yellow]")
                else:
                    print(f"🔄 Creating session {i+1}...")
                
                lock_system.lock()
                time.sleep(1)
                lock_system.unlock("session123")
                time.sleep(1)
            
            # แสดงประวัติ Session
            if RICH_AVAILABLE:
                self.console.print("\n📊 [bright_green]Session History:[/bright_green]")
            else:
                print("\n📊 Session History:")
            
            lock_system.show_sessions()
    
    def interactive_demo(self):
        """Demo แบบโต้ตอบ"""
        if not RICH_AVAILABLE:
            print("🎮 Interactive Demo")
            print("1. Lock/Unlock Demo")
            print("2. Monitoring Demo")
            print("3. Session Management Demo")
            print("4. Exit")
            
            while True:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "1":
                    self.demo_lock_unlock()
                elif choice == "2":
                    self.demo_monitoring()
                elif choice == "3":
                    self.demo_session_management()
                elif choice == "4":
                    break
                else:
                    print("Invalid choice!")
            return
        
        # Rich interactive demo
        while True:
            self.console.clear()
            
            # แสดงเมนูโต้ตอบ
            menu_table = Table(show_header=True, header_style="bold bright_white")
            menu_table.add_column("Option", style="bright_cyan")
            menu_table.add_column("Description", style="bright_white")
            menu_table.add_column("Status", style="bright_green")
            
            menu_options = [
                ("1", "🔐 Lock/Unlock Demo", "🎮 Interactive"),
                ("2", "📊 System Monitoring Demo", "📈 Real-time"),
                ("3", "📋 Session Management Demo", "📋 History"),
                ("4", "🌟 Show Features", "ℹ️ Information"),
                ("5", "🚪 Exit Demo", "👋 Goodbye")
            ]
            
            for option, description, status in menu_options:
                menu_table.add_row(option, description, status)
            
            menu_panel = Panel(
                menu_table,
                title="🎮 Interactive Demo Menu",
                box=ROUNDED,
                style="bright_blue"
            )
            
            self.console.print(menu_panel)
            
            choice = Prompt.ask("\n🎯 Select demo option", choices=["1", "2", "3", "4", "5"])
            
            if choice == "1":
                self.demo_lock_unlock()
            elif choice == "2":
                self.demo_monitoring()
            elif choice == "3":
                self.demo_session_management()
            elif choice == "4":
                self.show_features()
            elif choice == "5":
                break
            
            if choice != "5":
                input("\nPress Enter to continue...")
    
    def run_full_demo(self):
        """รันเดโมแบบเต็ม"""
        try:
            # แสดงแบนเนอร์
            self.show_banner()
            
            # แสดงฟีเจอร์
            self.show_features()
            
            if RICH_AVAILABLE:
                self.console.print("\n🎉 [bold bright_green]Welcome to Terminal Lock Demo![/bold bright_green]")
                time.sleep(2)
                
                # ถามว่าต้องการ Demo แบบไหน
                demo_type = Prompt.ask(
                    "\n🎯 Choose demo type",
                    choices=["auto", "interactive"],
                    default="auto"
                )
                
                if demo_type == "auto":
                    self.console.print("\n🚀 [bold bright_yellow]Starting Automatic Demo...[/bold bright_yellow]")
                    
                    # Demo อัตโนมัติ
                    self.demo_lock_unlock()
                    time.sleep(2)
                    
                    self.demo_monitoring()
                    time.sleep(2)
                    
                    self.demo_session_management()
                    
                    self.console.print("\n🎉 [bold bright_green]Automatic Demo Completed![/bold bright_green]")
                    
                else:
                    # Demo แบบโต้ตอบ
                    self.interactive_demo()
            else:
                # Demo แบบพื้นฐาน
                print("\n🎉 Welcome to Terminal Lock Demo!")
                print("Running basic demo...")
                
                self.demo_lock_unlock()
                print("\n📊 Monitoring Demo...")
                self.demo_monitoring()
                print("\n📋 Session Management Demo...")
                self.demo_session_management()
                
                print("\n🎉 Demo Completed!")
                
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                self.console.print("\n\n🛑 [bold bright_red]Demo interrupted by user[/bold bright_red]")
            else:
                print("\n\n🛑 Demo interrupted by user")
        
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"\n⚠️ [bold bright_red]Demo error: {e}[/bold bright_red]")
            else:
                print(f"\n⚠️ Demo error: {e}")
        
        finally:
            if RICH_AVAILABLE:
                self.console.print("\n👋 [bold bright_blue]Thank you for trying Terminal Lock Demo![/bold bright_blue]")
            else:
                print("\n👋 Thank you for trying Terminal Lock Demo!")


def main():
    """ฟังก์ชันหลักของ Demo"""
    demo = TerminalLockDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main() 