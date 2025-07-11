#!/usr/bin/env python3
"""
🔐 NICEGOLD Enterprise Terminal Lock System
Modern, Beautiful, and Secure Terminal Locking Solution

Version: 1.0 Enterprise Edition
Date: 11 July 2025
Status: Production Ready
"""

import os
import sys
import time
import json
import hashlib
import threading
import datetime
import platform
import getpass
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Rich library for beautiful terminal UI
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
    RICH_AVAILABLE = False
    print("⚠️  Rich library not available. Installing...")
    os.system("pip install rich")
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
        RICH_AVAILABLE = False

# Colorama for cross-platform color support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


@dataclass
class LockSession:
    """ข้อมูล Session การล็อค"""
    session_id: str
    user: str
    hostname: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    unlock_attempts: int = 0
    system_info: Dict[str, Any] = None
    security_events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.security_events is None:
            self.security_events = []
        if self.system_info is None:
            self.system_info = {}


class EnterpriseTerminalLock:
    """
    🔐 Enterprise Terminal Lock System
    ระบบล็อคเทอร์มินัลสำหรับองค์กรที่สวยงามและปลอดภัย
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """เริ่มต้นระบบล็อคเทอร์มินัล"""
        self.console = Console(force_terminal=True, width=120) if RICH_AVAILABLE else None
        self.config_path = config_path or "config/terminal_lock_config.json"
        self.lock_file = Path("temp/terminal.lock")
        self.session_file = Path("logs/terminal_lock_sessions.json")
        self.current_session: Optional[LockSession] = None
        self.is_locked = False
        self.lock_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
        
        # สร้างโฟลเดอร์ที่จำเป็น
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        
        # โหลดการตั้งค่า
        self.config = self._load_config()
        
        # ข้อมูลระบบ
        self.system_info = self._get_system_info()
        
        # สีและสไตล์สำหรับการแสดงผล
        self.colors = {
            'primary': '#00D4FF',      # Cyan สวยงาม
            'secondary': '#FF6B6B',    # Pink สดใส
            'success': '#51CF66',      # Green สำเร็จ
            'warning': '#FFD93D',      # Yellow เตือน
            'error': '#FF6B6B',        # Red ข้อผิดพลาด
            'info': '#74C0FC',         # Light blue ข้อมูล
            'dark': '#2C2C2C',         # Dark gray พื้นหลัง
            'light': '#F8F9FA',        # Light gray ข้อความ
            'gradient': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        }
        
        # ASCII Art สำหรับการแสดงผล
        self.ascii_art = {
            'lock': '''
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                                    🔐 LOCKED 🔐                                      ║
    ║                                                                                      ║
    ║  ███████╗███╗   ██╗████████╗███████╗██████╗ ██████╗ ██████╗ ██╗███████╗███████╗    ║
    ║  ██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝    ║
    ║  █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝██████╔╝██████╔╝██║███████╗█████╗      ║
    ║  ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔═══╝ ██╔══██╗██║╚════██║██╔══╝      ║
    ║  ███████╗██║ ╚████║   ██║   ███████╗██║  ██║██║     ██║  ██║██║███████║███████╗    ║
    ║  ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝    ║
    ║                                                                                      ║
    ║                            🏢 NICEGOLD ENTERPRISE                                   ║
    ║                          🔒 TERMINAL SECURITY SYSTEM                                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
            ''',
            'unlock': '''
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                                   🔓 UNLOCKED 🔓                                     ║
    ║                                                                                      ║
    ║  ██╗   ██╗███╗   ██╗██╗      ██████╗  ██████╗██╗  ██╗███████╗██████╗                ║
    ║  ██║   ██║████╗  ██║██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗               ║
    ║  ██║   ██║██╔██╗ ██║██║     ██║   ██║██║     █████╔╝ █████╗  ██║  ██║               ║
    ║  ██║   ██║██║╚██╗██║██║     ██║   ██║██║     ██╔═██╗ ██╔══╝  ██║  ██║               ║
    ║  ╚██████╔╝██║ ╚████║███████╗╚██████╔╝╚██████╗██║  ██╗███████╗██████╔╝               ║
    ║   ╚═════╝ ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝                ║
    ║                                                                                      ║
    ║                           ✅ ACCESS GRANTED                                          ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
            '''
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """โหลดการตั้งค่าระบบ"""
        default_config = {
            "security": {
                "max_unlock_attempts": 3,
                "auto_lock_timeout": 300,  # 5 นาที
                "require_password": True,
                "password_hash": None,
                "session_timeout": 3600,  # 1 ชั่วโมง
                "enable_monitoring": True,
                "log_attempts": True
            },
            "display": {
                "theme": "enterprise",
                "animation_speed": 0.05,
                "show_system_info": True,
                "show_real_time_stats": True,
                "refresh_interval": 1.0,
                "enable_gradient": True
            },
            "features": {
                "auto_screenshot": False,
                "remote_monitoring": False,
                "email_alerts": False,
                "webhook_notifications": False
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with default config
                self._merge_config(default_config, config)
                return default_config
            else:
                # สร้างไฟล์ config ใหม่
                Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            return default_config
    
    def _merge_config(self, base: Dict, update: Dict) -> None:
        """ผสานการตั้งค่า"""
        for key, value in update.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._merge_config(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value
    
    def _get_system_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลระบบ"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            
            return {
                'hostname': platform.node(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': cpu_percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_percent': memory.percent,
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_percent': (disk_usage.used / disk_usage.total) * 100,
                'uptime': time.time() - psutil.boot_time(),
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'hostname': platform.node(),
                'platform': platform.platform(),
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _generate_session_id(self) -> str:
        """สร้าง Session ID ที่ไม่ซ้ำกัน"""
        timestamp = datetime.datetime.now().isoformat()
        hostname = platform.node()
        user = getpass.getuser()
        combined = f"{timestamp}_{hostname}_{user}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _hash_password(self, password: str) -> str:
        """แฮชรหัสผ่าน"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str) -> bool:
        """ตรวจสอบรหัสผ่าน"""
        if not self.config['security']['password_hash']:
            return True  # ไม่มีรหัสผ่านตั้งไว้
        
        hashed = self._hash_password(password)
        return hashed == self.config['security']['password_hash']
    
    def set_password(self, password: str) -> None:
        """ตั้งรหัสผ่าน"""
        self.config['security']['password_hash'] = self._hash_password(password)
        self._save_config()
    
    def _save_config(self) -> None:
        """บันทึกการตั้งค่า"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error saving config: {e}")
    
    def _save_session(self) -> None:
        """บันทึกข้อมูล Session"""
        if not self.current_session:
            return
        
        try:
            sessions = []
            if self.session_file.exists():
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
            
            # แปลง datetime เป็น string
            session_data = asdict(self.current_session)
            session_data['start_time'] = self.current_session.start_time.isoformat()
            if self.current_session.end_time:
                session_data['end_time'] = self.current_session.end_time.isoformat()
            
            sessions.append(session_data)
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error saving session: {e}")
    
    def _show_welcome_screen(self) -> None:
        """แสดงหน้าจอต้อนรับ"""
        if not RICH_AVAILABLE:
            print("🔐 NICEGOLD Enterprise Terminal Lock System")
            print("=" * 60)
            return
        
        # Clear screen
        self.console.clear()
        
        # สร้าง Layout หลัก
        layout = Layout()
        
        # Header
        header = Panel(
            Align.center(
                Text("🔐 NICEGOLD Enterprise Terminal Lock System", 
                     style="bold bright_cyan"),
                vertical="middle"
            ),
            box=DOUBLE,
            style="bright_cyan",
            height=5
        )
        
        # System Info
        system_table = Table(show_header=True, header_style="bold bright_white")
        system_table.add_column("Property", style="bright_cyan")
        system_table.add_column("Value", style="bright_white")
        
        system_table.add_row("Hostname", self.system_info.get('hostname', 'Unknown'))
        system_table.add_row("Platform", self.system_info.get('platform', 'Unknown'))
        system_table.add_row("User", getpass.getuser())
        system_table.add_row("Session ID", self.current_session.session_id if self.current_session else 'None')
        system_table.add_row("Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        system_panel = Panel(
            system_table,
            title="🖥️  System Information",
            box=ROUNDED,
            style="bright_blue"
        )
        
        # Status
        status_text = Text("🔓 UNLOCKED", style="bold bright_green")
        if self.is_locked:
            status_text = Text("🔐 LOCKED", style="bold bright_red")
        
        status_panel = Panel(
            Align.center(status_text),
            title="Status",
            box=ROUNDED,
            style="bright_yellow"
        )
        
        # Layout setup
        layout.split_column(
            Layout(header, name="header", size=5),
            Layout(name="main", ratio=2),
            Layout(status_panel, name="status", size=5)
        )
        
        layout["main"].split_row(
            Layout(system_panel, name="system"),
            Layout(name="stats")
        )
        
        # Stats
        if self.config['display']['show_real_time_stats']:
            stats_table = Table(show_header=True, header_style="bold bright_white")
            stats_table.add_column("Metric", style="bright_cyan")
            stats_table.add_column("Value", style="bright_white")
            
            # CPU และ Memory
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            stats_table.add_row("CPU Usage", f"{cpu_percent:.1f}%")
            stats_table.add_row("Memory Usage", f"{memory.percent:.1f}%")
            stats_table.add_row("Available Memory", f"{memory.available / (1024**3):.1f} GB")
            stats_table.add_row("CPU Cores", str(psutil.cpu_count()))
            
            stats_panel = Panel(
                stats_table,
                title="📊 Real-time Statistics",
                box=ROUNDED,
                style="bright_magenta"
            )
            
            layout["stats"].update(stats_panel)
        
        # Display
        with Live(layout, refresh_per_second=2, console=self.console) as live:
            time.sleep(3)
    
    def _show_lock_screen(self) -> None:
        """แสดงหน้าจอล็อค"""
        if not RICH_AVAILABLE:
            print(self.ascii_art['lock'])
            return
        
        self.console.clear()
        
        # สร้าง animated lock screen
        lock_panel = Panel(
            Align.center(
                Text(self.ascii_art['lock'], style="bold bright_red"),
                vertical="middle"
            ),
            title="🔐 TERMINAL LOCKED",
            box=HEAVY,
            style="bright_red"
        )
        
        # ข้อมูลการล็อค
        lock_info = Table(show_header=True, header_style="bold bright_white")
        lock_info.add_column("Information", style="bright_yellow")
        lock_info.add_column("Value", style="bright_white")
        
        if self.current_session:
            lock_info.add_row("Session ID", self.current_session.session_id)
            lock_info.add_row("User", self.current_session.user)
            lock_info.add_row("Hostname", self.current_session.hostname)
            lock_info.add_row("Lock Time", self.current_session.start_time.strftime("%Y-%m-%d %H:%M:%S"))
            lock_info.add_row("Duration", str(datetime.datetime.now() - self.current_session.start_time))
            lock_info.add_row("Unlock Attempts", str(self.current_session.unlock_attempts))
        
        info_panel = Panel(
            lock_info,
            title="📋 Lock Information",
            box=ROUNDED,
            style="bright_blue"
        )
        
        # Real-time monitoring
        monitoring_panel = self._create_monitoring_panel()
        
        # Layout
        layout = Layout()
        layout.split_column(
            Layout(lock_panel, name="lock", size=15),
            Layout(name="main", ratio=2)
        )
        
        layout["main"].split_row(
            Layout(info_panel, name="info"),
            Layout(monitoring_panel, name="monitoring")
        )
        
        self.console.print(layout)
    
    def _create_monitoring_panel(self) -> Panel:
        """สร้างแผงการตรวจสอบระบบ"""
        if not RICH_AVAILABLE:
            return Panel("Monitoring not available")
        
        monitoring_table = Table(show_header=True, header_style="bold bright_white")
        monitoring_table.add_column("System Metric", style="bright_cyan")
        monitoring_table.add_column("Current Value", style="bright_white")
        monitoring_table.add_column("Status", style="bright_green")
        
        # ข้อมูลระบบล่าสุด
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            processes = len(psutil.pids())
            
            monitoring_table.add_row("CPU Usage", f"{cpu_percent:.1f}%", "🟢 Normal" if cpu_percent < 80 else "🔴 High")
            monitoring_table.add_row("Memory Usage", f"{memory.percent:.1f}%", "🟢 Normal" if memory.percent < 80 else "🔴 High")
            monitoring_table.add_row("Active Processes", str(processes), "🟢 Normal")
            monitoring_table.add_row("System Uptime", f"{psutil.boot_time():.0f}s", "🟢 Active")
            
        except Exception as e:
            monitoring_table.add_row("Error", str(e), "🔴 Error")
        
        return Panel(
            monitoring_table,
            title="📊 System Monitoring",
            box=ROUNDED,
            style="bright_magenta"
        )
    
    def _animate_lock(self) -> None:
        """แอนิเมชันการล็อค"""
        if not RICH_AVAILABLE:
            print("🔐 Locking...")
            return
        
        with self.console.status("[bold bright_yellow]Locking terminal...", spinner="dots") as status:
            for i in range(3):
                time.sleep(0.5)
                status.update(f"[bold bright_yellow]Locking{'.' * (i + 1)}")
            
            status.update("[bold bright_red]🔐 LOCKED!")
            time.sleep(1)
    
    def _animate_unlock(self) -> None:
        """แอนิเมชันการปลดล็อค"""
        if not RICH_AVAILABLE:
            print("🔓 Unlocking...")
            return
        
        with self.console.status("[bold bright_green]Unlocking terminal...", spinner="dots") as status:
            for i in range(3):
                time.sleep(0.5)
                status.update(f"[bold bright_green]Unlocking{'.' * (i + 1)}")
            
            status.update("[bold bright_green]🔓 UNLOCKED!")
            time.sleep(1)
    
    def _monitoring_loop(self) -> None:
        """วนรอบการตรวจสอบระบบ"""
        while not self.stop_monitoring and self.is_locked:
            try:
                # อัพเดทข้อมูลระบบ
                self.system_info = self._get_system_info()
                
                # บันทึกข้อมูลการตรวจสอบ
                if self.current_session:
                    monitoring_data = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'system_info': self.system_info,
                        'session_id': self.current_session.session_id
                    }
                    self.current_session.security_events.append(monitoring_data)
                
                # หน่วงเวลาตามการตั้งค่า
                time.sleep(self.config['display']['refresh_interval'])
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(5)
    
    def lock(self) -> None:
        """ล็อคเทอร์มินัล"""
        if self.is_locked:
            print("⚠️  Terminal is already locked!")
            return
        
        # แสดงหน้าจอต้อนรับ
        self._show_welcome_screen()
        
        # สร้าง Session ใหม่
        self.current_session = LockSession(
            session_id=self._generate_session_id(),
            user=getpass.getuser(),
            hostname=platform.node(),
            start_time=datetime.datetime.now(),
            system_info=self.system_info
        )
        
        # สร้างไฟล์ล็อค
        self.lock_file.touch()
        with open(self.lock_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.current_session), f, default=str, indent=2)
        
        # ตั้งค่าสถานะ
        self.is_locked = True
        
        # แอนิเมชันการล็อค
        self._animate_lock()
        
        # เริ่มการตรวจสอบระบบ
        if self.config['security']['enable_monitoring']:
            self.stop_monitoring = False
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
        
        # บันทึก Session
        self._save_session()
        
        print("🔐 Terminal locked successfully!")
        if RICH_AVAILABLE:
            self.console.print("🔐 [bold bright_red]Terminal locked successfully![/bold bright_red]")
    
    def unlock(self, password: Optional[str] = None) -> bool:
        """ปลดล็อคเทอร์มินัล"""
        if not self.is_locked:
            print("⚠️  Terminal is not locked!")
            return True
        
        # แสดงหน้าจอล็อค
        self._show_lock_screen()
        
        # ตรวจสอบรหัสผ่าน
        if self.config['security']['require_password']:
            if not password:
                if RICH_AVAILABLE:
                    password = Prompt.ask("🔑 Enter password to unlock", password=True)
                else:
                    password = getpass.getpass("🔑 Enter password to unlock: ")
            
            # เพิ่มจำนวนครั้งที่พยายามปลดล็อค
            if self.current_session:
                self.current_session.unlock_attempts += 1
            
            if not self._verify_password(password):
                if RICH_AVAILABLE:
                    self.console.print("❌ [bold bright_red]Invalid password![/bold bright_red]")
                else:
                    print("❌ Invalid password!")
                
                # ตรวจสอบจำนวนครั้งที่พยายาม
                if (self.current_session and 
                    self.current_session.unlock_attempts >= self.config['security']['max_unlock_attempts']):
                    
                    if RICH_AVAILABLE:
                        self.console.print("🚨 [bold bright_red]Maximum unlock attempts reached![/bold bright_red]")
                    else:
                        print("🚨 Maximum unlock attempts reached!")
                    
                    # บันทึกเหตุการณ์ความปลอดภัย
                    if self.current_session:
                        security_event = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'event': 'max_unlock_attempts_reached',
                            'attempts': self.current_session.unlock_attempts
                        }
                        self.current_session.security_events.append(security_event)
                    
                    return False
                
                return False
        
        # ปลดล็อคสำเร็จ
        self.is_locked = False
        self.stop_monitoring = True
        
        # อัพเดท Session
        if self.current_session:
            self.current_session.end_time = datetime.datetime.now()
        
        # ลบไฟล์ล็อค
        if self.lock_file.exists():
            self.lock_file.unlink()
        
        # แอนิเมชันปลดล็อค
        self._animate_unlock()
        
        # บันทึก Session
        self._save_session()
        
        # แสดงหน้าจอปลดล็อค
        if RICH_AVAILABLE:
            unlock_panel = Panel(
                Align.center(
                    Text(self.ascii_art['unlock'], style="bold bright_green"),
                    vertical="middle"
                ),
                title="🔓 TERMINAL UNLOCKED",
                box=HEAVY,
                style="bright_green"
            )
            self.console.print(unlock_panel)
        else:
            print(self.ascii_art['unlock'])
        
        print("🔓 Terminal unlocked successfully!")
        return True
    
    def status(self) -> Dict[str, Any]:
        """ดูสถานะการล็อค"""
        return {
            'is_locked': self.is_locked,
            'current_session': asdict(self.current_session) if self.current_session else None,
            'lock_file_exists': self.lock_file.exists(),
            'system_info': self.system_info,
            'config': self.config
        }
    
    def show_sessions(self) -> None:
        """แสดงประวัติ Session"""
        if not self.session_file.exists():
            print("📋 No sessions found.")
            return
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            if not RICH_AVAILABLE:
                print("📋 Terminal Lock Sessions:")
                print("-" * 50)
                for session in sessions[-10:]:  # แสดง 10 session ล่าสุด
                    print(f"Session ID: {session['session_id']}")
                    print(f"User: {session['user']}")
                    print(f"Start: {session['start_time']}")
                    print(f"End: {session.get('end_time', 'Not ended')}")
                    print(f"Attempts: {session['unlock_attempts']}")
                    print("-" * 50)
                return
            
            # แสดงด้วย Rich
            sessions_table = Table(show_header=True, header_style="bold bright_white")
            sessions_table.add_column("Session ID", style="bright_cyan")
            sessions_table.add_column("User", style="bright_white")
            sessions_table.add_column("Hostname", style="bright_blue")
            sessions_table.add_column("Start Time", style="bright_green")
            sessions_table.add_column("End Time", style="bright_yellow")
            sessions_table.add_column("Duration", style="bright_magenta")
            sessions_table.add_column("Attempts", style="bright_red")
            
            for session in sessions[-10:]:  # แสดง 10 session ล่าสุด
                start_time = datetime.datetime.fromisoformat(session['start_time'])
                end_time = session.get('end_time')
                
                if end_time:
                    end_time = datetime.datetime.fromisoformat(end_time)
                    duration = str(end_time - start_time)
                    end_time_str = end_time.strftime("%H:%M:%S")
                else:
                    duration = "Ongoing"
                    end_time_str = "N/A"
                
                sessions_table.add_row(
                    session['session_id'][:8],
                    session['user'],
                    session['hostname'],
                    start_time.strftime("%H:%M:%S"),
                    end_time_str,
                    duration,
                    str(session['unlock_attempts'])
                )
            
            sessions_panel = Panel(
                sessions_table,
                title="📋 Terminal Lock Sessions (Last 10)",
                box=ROUNDED,
                style="bright_blue"
            )
            
            self.console.print(sessions_panel)
            
        except Exception as e:
            print(f"⚠️  Error loading sessions: {e}")
    
    def cleanup(self) -> None:
        """ทำความสะอาดระบบ"""
        # หยุดการตรวจสอบ
        self.stop_monitoring = True
        
        # ปลดล็อคถ้าล็อคอยู่
        if self.is_locked:
            self.is_locked = False
            if self.current_session:
                self.current_session.end_time = datetime.datetime.now()
                self._save_session()
        
        # ลบไฟล์ล็อค
        if self.lock_file.exists():
            self.lock_file.unlink()
        
        print("🧹 Terminal lock system cleaned up.")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def main():
    """ฟังก์ชันหลักสำหรับทดสอบระบบ"""
    print("🔐 NICEGOLD Enterprise Terminal Lock System")
    print("=" * 60)
    
    # สร้างระบบล็อค
    with EnterpriseTerminalLock() as lock_system:
        while True:
            print("\n📋 Available Commands:")
            print("1. 🔐 Lock Terminal")
            print("2. 🔓 Unlock Terminal")
            print("3. 📊 Show Status")
            print("4. 📋 Show Sessions")
            print("5. 🔑 Set Password")
            print("6. 🚪 Exit")
            
            try:
                choice = input("\n🎯 Enter your choice: ").strip()
                
                if choice == "1":
                    lock_system.lock()
                elif choice == "2":
                    lock_system.unlock()
                elif choice == "3":
                    status = lock_system.status()
                    print("\n📊 System Status:")
                    print(f"Locked: {status['is_locked']}")
                    print(f"Lock File: {status['lock_file_exists']}")
                    if status['current_session']:
                        print(f"Session ID: {status['current_session']['session_id']}")
                elif choice == "4":
                    lock_system.show_sessions()
                elif choice == "5":
                    password = getpass.getpass("🔑 Enter new password: ")
                    confirm = getpass.getpass("🔄 Confirm password: ")
                    if password == confirm:
                        lock_system.set_password(password)
                        print("✅ Password set successfully!")
                    else:
                        print("❌ Passwords do not match!")
                elif choice == "6":
                    break
                else:
                    print("❌ Invalid choice!")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main() 