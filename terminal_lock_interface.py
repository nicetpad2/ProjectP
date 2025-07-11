#!/usr/bin/env python3
"""
🔐 NICEGOLD Enterprise Terminal Lock Interface
Simple and Beautiful Terminal Lock Integration

Version: 1.0 Enterprise Edition
Date: 11 July 2025
Status: Production Ready
"""

import os
import sys
import json
import time
import hashlib
import getpass
import platform
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# สีสำหรับ Terminal (Cross-platform)
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback สำหรับกรณีที่ไม่มี colorama
    class Fore:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        RESET = '\033[0m'
    
    class Back:
        RED = '\033[101m'
        GREEN = '\033[102m'
        YELLOW = '\033[103m'
        BLUE = '\033[104m'
        MAGENTA = '\033[105m'
        CYAN = '\033[106m'
        WHITE = '\033[107m'
        RESET = '\033[0m'
    
    class Style:
        BRIGHT = '\033[1m'
        DIM = '\033[2m'
        RESET_ALL = '\033[0m'


class SimpleTerminalLock:
    """
    🔐 Simple Terminal Lock System
    Beautiful, Secure, and Easy to Use
    """
    
    def __init__(self, config_file: str = "config/terminal_lock_config.json"):
        self.config_file = config_file
        self.lock_file = Path("temp/terminal_simple.lock")
        self.is_locked = False
        self.session_start = None
        self.unlock_attempts = 0
        self.max_attempts = 3
        self.password_hash = None
        
        # สร้างโฟลเดอร์ที่จำเป็น
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # โหลดการตั้งค่า
        self.load_config()
        
        # ASCII Art
        self.lock_art = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                    {Fore.RED}🔐 LOCKED 🔐{Fore.CYAN}                                      ║
║                                                                                      ║
║  {Fore.YELLOW}███████╗███╗   ██╗████████╗███████╗██████╗ ██████╗ ██████╗ ██╗███████╗███████╗{Fore.CYAN}    ║
║  {Fore.YELLOW}██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝{Fore.CYAN}    ║
║  {Fore.YELLOW}█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝██████╔╝██████╔╝██║███████╗█████╗{Fore.CYAN}      ║
║  {Fore.YELLOW}██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔═══╝ ██╔══██╗██║╚════██║██╔══╝{Fore.CYAN}      ║
║  {Fore.YELLOW}███████╗██║ ╚████║   ██║   ███████╗██║  ██║██║     ██║  ██║██║███████║███████╗{Fore.CYAN}    ║
║  {Fore.YELLOW}╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝{Fore.CYAN}    ║
║                                                                                      ║
║                            {Fore.MAGENTA}🏢 NICEGOLD ENTERPRISE{Fore.CYAN}                                   ║
║                          {Fore.MAGENTA}🔒 TERMINAL SECURITY SYSTEM{Fore.CYAN}                                ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
        
        self.unlock_art = f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                   {Fore.GREEN}🔓 UNLOCKED 🔓{Fore.GREEN}                                     ║
║                                                                                      ║
║  {Fore.CYAN}██╗   ██╗███╗   ██╗██╗      ██████╗  ██████╗██╗  ██╗███████╗██████╗{Fore.GREEN}                ║
║  {Fore.CYAN}██║   ██║████╗  ██║██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗{Fore.GREEN}               ║
║  {Fore.CYAN}██║   ██║██╔██╗ ██║██║     ██║   ██║██║     █████╔╝ █████╗  ██║  ██║{Fore.GREEN}               ║
║  {Fore.CYAN}██║   ██║██║╚██╗██║██║     ██║   ██║██║     ██╔═██╗ ██╔══╝  ██║  ██║{Fore.GREEN}               ║
║  {Fore.CYAN}╚██████╔╝██║ ╚████║███████╗╚██████╔╝╚██████╗██║  ██╗███████╗██████╔╝{Fore.GREEN}               ║
║  {Fore.CYAN}╚═════╝ ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝{Fore.GREEN}                ║
║                                                                                      ║
║                           {Fore.YELLOW}✅ ACCESS GRANTED{Fore.GREEN}                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
        
        self.banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  {Fore.YELLOW}🔐 NICEGOLD ENTERPRISE TERMINAL LOCK SYSTEM{Fore.CYAN}                                          ║
║                                                                                      ║
║  {Fore.MAGENTA}████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗{Fore.CYAN}                      ║
║  {Fore.MAGENTA}╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║{Fore.CYAN}                      ║
║  {Fore.MAGENTA}   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║{Fore.CYAN}                      ║
║  {Fore.MAGENTA}   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║{Fore.CYAN}                      ║
║  {Fore.MAGENTA}   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗{Fore.CYAN}                 ║
║  {Fore.MAGENTA}   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝{Fore.CYAN}                 ║
║                                                                                      ║
║  {Fore.GREEN}██╗      ██████╗  ██████╗██╗  ██╗    ██████╗ ███████╗███╗   ███╗ ██████╗{Fore.CYAN}          ║
║  {Fore.GREEN}██║     ██╔═══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗{Fore.CYAN}         ║
║  {Fore.GREEN}██║     ██║   ██║██║     █████╔╝     ██║  ██║█████╗  ██╔████╔██║██║   ██║{Fore.CYAN}         ║
║  {Fore.GREEN}██║     ██║   ██║██║     ██╔═██╗     ██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║{Fore.CYAN}         ║
║  {Fore.GREEN}███████╗╚██████╔╝╚██████╗██║  ██╗    ██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝{Fore.CYAN}         ║
║  {Fore.GREEN}╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝{Fore.CYAN}          ║
║                                                                                      ║
║                           {Fore.YELLOW}🚀 Enterprise Production Ready{Fore.CYAN}                             ║
║                              {Fore.YELLOW}✨ Beautiful & Modern{Fore.CYAN}                                  ║
║                               {Fore.YELLOW}🔐 Security First{Fore.CYAN}                                     ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
    
    def load_config(self) -> None:
        """โหลดการตั้งค่า"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.max_attempts = config.get('security', {}).get('max_unlock_attempts', 3)
                    self.password_hash = config.get('security', {}).get('password_hash')
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Warning: Could not load config: {e}{Style.RESET_ALL}")
    
    def save_config(self) -> None:
        """บันทึกการตั้งค่า"""
        try:
            config = {
                "security": {
                    "max_unlock_attempts": self.max_attempts,
                    "password_hash": self.password_hash
                }
            }
            
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Warning: Could not save config: {e}{Style.RESET_ALL}")
    
    def hash_password(self, password: str) -> str:
        """แฮชรหัสผ่าน"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def set_password(self, password: str) -> None:
        """ตั้งรหัสผ่าน"""
        self.password_hash = self.hash_password(password)
        self.save_config()
        print(f"{Fore.GREEN}✅ Password set successfully!{Style.RESET_ALL}")
    
    def verify_password(self, password: str) -> bool:
        """ตรวจสอบรหัสผ่าน"""
        if not self.password_hash:
            return True  # ไม่มีรหัสผ่าน
        return self.hash_password(password) == self.password_hash
    
    def clear_screen(self) -> None:
        """ล้างหน้าจอ"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_banner(self) -> None:
        """แสดงแบนเนอร์"""
        self.clear_screen()
        print(self.banner)
        time.sleep(1)
    
    def show_system_info(self) -> None:
        """แสดงข้อมูลระบบ"""
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                                {Fore.YELLOW}📊 SYSTEM INFORMATION{Fore.CYAN}                                ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  {Fore.WHITE}🖥️  Hostname:{Fore.CYAN} {platform.node():<62} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}💻 Platform:{Fore.CYAN} {platform.platform():<63} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}👤 User:{Fore.CYAN} {getpass.getuser():<68} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}📅 Date:{Fore.CYAN} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<68} {Fore.CYAN}║")
        
        if self.is_locked:
            duration = datetime.datetime.now() - self.session_start
            print(f"║  {Fore.WHITE}🔐 Lock Duration:{Fore.CYAN} {str(duration):<58} {Fore.CYAN}║")
            print(f"║  {Fore.WHITE}🔢 Unlock Attempts:{Fore.CYAN} {self.unlock_attempts:<56} {Fore.CYAN}║")
        
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    def animate_locking(self) -> None:
        """แอนิเมชันการล็อค"""
        print(f"{Fore.YELLOW}🔐 Locking terminal", end="")
        for i in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print(f" {Fore.RED}LOCKED!{Style.RESET_ALL}")
        time.sleep(1)
    
    def animate_unlocking(self) -> None:
        """แอนิเมชันการปลดล็อค"""
        print(f"{Fore.GREEN}🔓 Unlocking terminal", end="")
        for i in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print(f" {Fore.GREEN}UNLOCKED!{Style.RESET_ALL}")
        time.sleep(1)
    
    def lock(self) -> None:
        """ล็อคเทอร์มินัล"""
        if self.is_locked:
            print(f"{Fore.YELLOW}⚠️  Terminal is already locked!{Style.RESET_ALL}")
            return
        
        self.show_banner()
        self.animate_locking()
        
        # ตั้งค่าการล็อค
        self.is_locked = True
        self.session_start = datetime.datetime.now()
        self.unlock_attempts = 0
        
        # สร้างไฟล์ล็อค
        with open(self.lock_file, 'w', encoding='utf-8') as f:
            json.dump({
                'locked': True,
                'start_time': self.session_start.isoformat(),
                'hostname': platform.node(),
                'user': getpass.getuser()
            }, f, indent=2)
        
        print(f"{Fore.GREEN}✅ Terminal locked successfully!{Style.RESET_ALL}")
    
    def unlock(self, password: str = None) -> bool:
        """ปลดล็อคเทอร์มินัล"""
        if not self.is_locked:
            print(f"{Fore.YELLOW}⚠️  Terminal is not locked!{Style.RESET_ALL}")
            return True
        
        # แสดงหน้าจอล็อค
        self.clear_screen()
        print(self.lock_art)
        self.show_system_info()
        
        # ตรวจสอบรหัสผ่าน
        if self.password_hash:
            if not password:
                password = getpass.getpass(f"{Fore.CYAN}🔑 Enter password to unlock: {Style.RESET_ALL}")
            
            self.unlock_attempts += 1
            
            if not self.verify_password(password):
                print(f"{Fore.RED}❌ Invalid password! (Attempt {self.unlock_attempts}/{self.max_attempts}){Style.RESET_ALL}")
                
                if self.unlock_attempts >= self.max_attempts:
                    print(f"{Fore.RED}🚨 Maximum unlock attempts reached!{Style.RESET_ALL}")
                    return False
                
                return False
        
        # ปลดล็อคสำเร็จ
        self.animate_unlocking()
        
        self.is_locked = False
        
        # ลบไฟล์ล็อค
        if self.lock_file.exists():
            self.lock_file.unlink()
        
        # แสดงหน้าจอปลดล็อค
        self.clear_screen()
        print(self.unlock_art)
        print(f"{Fore.GREEN}✅ Terminal unlocked successfully!{Style.RESET_ALL}")
        
        return True
    
    def status(self) -> Dict[str, Any]:
        """ดูสถานะ"""
        return {
            'is_locked': self.is_locked,
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'unlock_attempts': self.unlock_attempts,
            'max_attempts': self.max_attempts,
            'has_password': self.password_hash is not None,
            'lock_file_exists': self.lock_file.exists()
        }
    
    def show_status(self) -> None:
        """แสดงสถานะ"""
        status = self.status()
        
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                                {Fore.YELLOW}📊 LOCK STATUS{Fore.CYAN}                                       ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  {Fore.WHITE}🔐 Status:{Fore.CYAN} {'LOCKED' if status['is_locked'] else 'UNLOCKED':<65} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}📁 Lock File:{Fore.CYAN} {'EXISTS' if status['lock_file_exists'] else 'NOT FOUND':<62} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}🔑 Password:{Fore.CYAN} {'SET' if status['has_password'] else 'NOT SET':<64} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}🔢 Attempts:{Fore.CYAN} {status['unlock_attempts']}/{status['max_attempts']:<64} {Fore.CYAN}║")
        
        if status['session_start']:
            print(f"║  {Fore.WHITE}📅 Session Start:{Fore.CYAN} {status['session_start']:<59} {Fore.CYAN}║")
        
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    def interactive_menu(self) -> None:
        """เมนูโต้ตอบ"""
        while True:
            self.clear_screen()
            print(self.banner)
            
            print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
            print(f"║                                {Fore.YELLOW}📋 TERMINAL LOCK MENU{Fore.CYAN}                                ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  {Fore.WHITE}1.{Fore.CYAN} 🔐 Lock Terminal                                                            ║")
            print(f"║  {Fore.WHITE}2.{Fore.CYAN} 🔓 Unlock Terminal                                                          ║")
            print(f"║  {Fore.WHITE}3.{Fore.CYAN} 📊 Show Status                                                             ║")
            print(f"║  {Fore.WHITE}4.{Fore.CYAN} 🔑 Set Password                                                            ║")
            print(f"║  {Fore.WHITE}5.{Fore.CYAN} 🖥️  Show System Info                                                       ║")
            print(f"║  {Fore.WHITE}6.{Fore.CYAN} 🚪 Exit                                                                    ║")
            print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            
            try:
                choice = input(f"{Fore.YELLOW}🎯 Enter your choice (1-6): {Style.RESET_ALL}").strip()
                
                if choice == "1":
                    self.lock()
                elif choice == "2":
                    self.unlock()
                elif choice == "3":
                    self.show_status()
                elif choice == "4":
                    password = getpass.getpass(f"{Fore.CYAN}🔑 Enter new password: {Style.RESET_ALL}")
                    confirm = getpass.getpass(f"{Fore.CYAN}🔄 Confirm password: {Style.RESET_ALL}")
                    if password == confirm:
                        self.set_password(password)
                    else:
                        print(f"{Fore.RED}❌ Passwords do not match!{Style.RESET_ALL}")
                elif choice == "5":
                    self.show_system_info()
                elif choice == "6":
                    print(f"{Fore.GREEN}👋 Goodbye!{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}❌ Invalid choice!{Style.RESET_ALL}")
                
                if choice != "6":
                    input(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}🛑 Interrupted by user{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}⚠️  Error: {e}{Style.RESET_ALL}")
                input(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")


def main():
    """ฟังก์ชันหลัก"""
    print(f"{Fore.CYAN}🔐 NICEGOLD Enterprise Terminal Lock System{Style.RESET_ALL}")
    print(f"{Fore.WHITE}=" * 60 + Style.RESET_ALL)
    
    # สร้าง Terminal Lock
    lock = SimpleTerminalLock()
    
    # ตรวจสอบการใช้งาน
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "lock":
            lock.lock()
        elif command == "unlock":
            password = sys.argv[2] if len(sys.argv) > 2 else None
            lock.unlock(password)
        elif command == "status":
            lock.show_status()
        elif command == "demo":
            # Quick demo
            print(f"{Fore.YELLOW}🎮 Running Demo...{Style.RESET_ALL}")
            lock.set_password("demo123")
            lock.lock()
            time.sleep(3)
            lock.unlock("demo123")
        else:
            print(f"{Fore.RED}❌ Unknown command: {command}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Usage: python terminal_lock_interface.py [lock|unlock|status|demo]{Style.RESET_ALL}")
    else:
        # เรียกใช้เมนูโต้ตอบ
        lock.interactive_menu()


if __name__ == "__main__":
    main() 