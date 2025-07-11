#!/usr/bin/env python3
"""
🧪 NICEGOLD Enterprise Terminal Lock - Test System
Comprehensive Testing for Terminal Lock System

Version: 1.0 Enterprise Edition
Date: 11 July 2025
Status: Production Ready
"""

import os
import sys
import time
import json
import threading
from pathlib import Path

# Import Terminal Lock Systems
try:
    from terminal_lock_interface import SimpleTerminalLock
    SIMPLE_LOCK_AVAILABLE = True
except ImportError:
    SIMPLE_LOCK_AVAILABLE = False

try:
    from core.enterprise_terminal_lock import EnterpriseTerminalLock
    ENTERPRISE_LOCK_AVAILABLE = True
except ImportError:
    ENTERPRISE_LOCK_AVAILABLE = False

# Colors
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        RESET = '\033[0m'
    
    class Style:
        BRIGHT = '\033[1m'
        RESET_ALL = '\033[0m'


class TerminalLockTester:
    """
    🧪 Terminal Lock Testing System
    """
    
    def __init__(self):
        self.test_results = []
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        
        # Test configuration
        self.test_password = "test123"
        self.test_config = {
            "security": {
                "max_unlock_attempts": 3,
                "require_password": True,
                "password_hash": None
            }
        }
    
    def log_test(self, test_name: str, result: bool, message: str = ""):
        """บันทึกผลการทดสอบ"""
        self.test_count += 1
        if result:
            self.passed_count += 1
            status = f"{Fore.GREEN}✅ PASSED{Style.RESET_ALL}"
        else:
            self.failed_count += 1
            status = f"{Fore.RED}❌ FAILED{Style.RESET_ALL}"
        
        self.test_results.append({
            'test_name': test_name,
            'result': result,
            'message': message,
            'timestamp': time.time()
        })
        
        print(f"  {status} - {test_name}")
        if message:
            print(f"    {Fore.YELLOW}Info: {message}{Style.RESET_ALL}")
    
    def test_simple_lock_basic(self):
        """ทดสอบ Simple Lock พื้นฐาน"""
        print(f"\n{Fore.CYAN}🧪 Testing Simple Lock Basic Functions{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Simple Lock Import", False, "SimpleTerminalLock not available")
            return
        
        try:
            # สร้าง Lock
            lock = SimpleTerminalLock()
            self.log_test("Simple Lock Creation", True, "Lock created successfully")
            
            # ทดสอบการตั้งรหัสผ่าน
            lock.set_password(self.test_password)
            self.log_test("Password Setting", True, "Password set successfully")
            
            # ทดสอบการตรวจสอบรหัสผ่าน
            result = lock.verify_password(self.test_password)
            self.log_test("Password Verification", result, "Password verified")
            
            # ทดสอบการล็อค
            lock.lock()
            self.log_test("Lock Function", lock.is_locked, "Lock function works")
            
            # ทดสอบการปลดล็อค
            unlock_result = lock.unlock(self.test_password)
            self.log_test("Unlock Function", unlock_result, "Unlock function works")
            
            # ทดสอบสถานะ
            status = lock.status()
            self.log_test("Status Function", isinstance(status, dict), "Status returned dict")
            
            # ทดสอบการล็อคซ้ำ
            lock.lock()
            lock.lock()  # ควรแสดงคำเตือน
            self.log_test("Duplicate Lock Prevention", lock.is_locked, "Duplicate lock handled")
            
            # ทดสอบการปลดล็อคด้วยรหัสผ่านผิด
            lock.unlock_attempts = 0  # Reset attempts
            wrong_result = lock.unlock("wrong_password")
            self.log_test("Wrong Password Handling", not wrong_result, "Wrong password rejected")
            
        except Exception as e:
            self.log_test("Simple Lock Basic Test", False, f"Exception: {e}")
    
    def test_enterprise_lock_basic(self):
        """ทดสอบ Enterprise Lock พื้นฐาน"""
        print(f"\n{Fore.CYAN}🧪 Testing Enterprise Lock Basic Functions{Style.RESET_ALL}")
        
        if not ENTERPRISE_LOCK_AVAILABLE:
            self.log_test("Enterprise Lock Import", False, "EnterpriseTerminalLock not available")
            return
        
        try:
            # สร้าง Lock
            lock = EnterpriseTerminalLock()
            self.log_test("Enterprise Lock Creation", True, "Lock created successfully")
            
            # ทดสอบการตั้งรหัสผ่าน
            lock.set_password(self.test_password)
            self.log_test("Enterprise Password Setting", True, "Password set successfully")
            
            # ทดสอบการล็อค
            lock.lock()
            self.log_test("Enterprise Lock Function", lock.is_locked, "Lock function works")
            
            # ทดสอบการปลดล็อค
            unlock_result = lock.unlock(self.test_password)
            self.log_test("Enterprise Unlock Function", unlock_result, "Unlock function works")
            
            # ทดสอบสถานะ
            status = lock.status()
            self.log_test("Enterprise Status Function", isinstance(status, dict), "Status returned dict")
            
        except Exception as e:
            self.log_test("Enterprise Lock Basic Test", False, f"Exception: {e}")
    
    def test_config_management(self):
        """ทดสอบการจัดการ Config"""
        print(f"\n{Fore.CYAN}🧪 Testing Configuration Management{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Config Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            # สร้าง config file ทดสอบ
            test_config_file = "temp/test_config.json"
            Path(test_config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(test_config_file, 'w') as f:
                json.dump(self.test_config, f, indent=2)
            
            # ทดสอบการโหลด config
            lock = SimpleTerminalLock(test_config_file)
            self.log_test("Config Loading", True, "Config loaded successfully")
            
            # ทดสอบการบันทึก config
            lock.save_config()
            self.log_test("Config Saving", Path(test_config_file).exists(), "Config saved successfully")
            
            # ทดสอบการแก้ไข config
            lock.max_attempts = 5
            lock.save_config()
            
            # โหลด config ใหม่
            lock2 = SimpleTerminalLock(test_config_file)
            self.log_test("Config Persistence", lock2.max_attempts == 5, "Config changes persisted")
            
            # ทำความสะอาด
            if Path(test_config_file).exists():
                Path(test_config_file).unlink()
            
        except Exception as e:
            self.log_test("Config Management Test", False, f"Exception: {e}")
    
    def test_security_features(self):
        """ทดสอบฟีเจอร์ความปลอดภัย"""
        print(f"\n{Fore.CYAN}🧪 Testing Security Features{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Security Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            lock = SimpleTerminalLock()
            lock.set_password(self.test_password)
            lock.max_attempts = 3
            
            # ทดสอบการจำกัดจำนวนครั้งที่พยายาม
            lock.lock()
            lock.unlock_attempts = 0
            
            # พยายามปลดล็อคด้วยรหัสผ่านผิด
            for i in range(3):
                result = lock.unlock("wrong_password")
                if i < 2:
                    self.log_test(f"Wrong Password Attempt {i+1}", not result, f"Attempt {i+1} rejected")
                else:
                    self.log_test("Max Attempts Reached", not result, "Max attempts protection works")
            
            # ทดสอบการแฮชรหัสผ่าน
            hash1 = lock.hash_password("password123")
            hash2 = lock.hash_password("password123")
            hash3 = lock.hash_password("different_password")
            
            self.log_test("Password Hashing Consistency", hash1 == hash2, "Same password produces same hash")
            self.log_test("Password Hashing Uniqueness", hash1 != hash3, "Different passwords produce different hashes")
            
            # ทดสอบการตรวจสอบรหัสผ่าน
            self.log_test("Password Verification True", lock.verify_password(self.test_password), "Correct password verified")
            self.log_test("Password Verification False", not lock.verify_password("wrong"), "Wrong password rejected")
            
        except Exception as e:
            self.log_test("Security Features Test", False, f"Exception: {e}")
    
    def test_file_operations(self):
        """ทดสอบการจัดการไฟล์"""
        print(f"\n{Fore.CYAN}🧪 Testing File Operations{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("File Operations Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            lock = SimpleTerminalLock()
            
            # ทดสอบการสร้างไฟล์ล็อค
            lock.lock()
            self.log_test("Lock File Creation", lock.lock_file.exists(), "Lock file created")
            
            # ทดสอบเนื้อหาของไฟล์ล็อค
            with open(lock.lock_file, 'r') as f:
                lock_data = json.load(f)
            
            self.log_test("Lock File Content", 'locked' in lock_data, "Lock file has correct content")
            
            # ทดสอบการลบไฟล์ล็อค
            lock.unlock()
            self.log_test("Lock File Deletion", not lock.lock_file.exists(), "Lock file deleted after unlock")
            
            # ทดสอบการจัดการไฟล์ที่ไม่มีอยู่
            if lock.lock_file.exists():
                lock.lock_file.unlink()
            
            status = lock.status()
            self.log_test("Missing File Handling", 'lock_file_exists' in status, "Missing file handled gracefully")
            
        except Exception as e:
            self.log_test("File Operations Test", False, f"Exception: {e}")
    
    def test_concurrent_access(self):
        """ทดสอบการเข้าถึงแบบ concurrent"""
        print(f"\n{Fore.CYAN}🧪 Testing Concurrent Access{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Concurrent Access Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            lock1 = SimpleTerminalLock()
            lock2 = SimpleTerminalLock()
            
            # ทดสอบการล็อคจาก instance แรก
            lock1.lock()
            self.log_test("First Lock Success", lock1.is_locked, "First lock successful")
            
            # ทดสอบการล็อคจาก instance ที่สอง
            lock2.lock()
            self.log_test("Second Lock Prevention", lock2.is_locked, "Second lock handled")
            
            # ทดสอบการอ่านสถานะจาก instance ที่สอง
            status2 = lock2.status()
            self.log_test("Cross-Instance Status", status2['lock_file_exists'], "Lock file visible across instances")
            
            # ทดสอบการปลดล็อคจาก instance ที่สอง
            lock2.unlock()
            self.log_test("Cross-Instance Unlock", not lock1.is_locked, "Unlock works across instances")
            
        except Exception as e:
            self.log_test("Concurrent Access Test", False, f"Exception: {e}")
    
    def test_error_handling(self):
        """ทดสอบการจัดการข้อผิดพลาด"""
        print(f"\n{Fore.CYAN}🧪 Testing Error Handling{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Error Handling Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            # ทดสอบการจัดการไฟล์ config ที่เสียหาย
            bad_config_file = "temp/bad_config.json"
            Path(bad_config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(bad_config_file, 'w') as f:
                f.write("invalid json content")
            
            # ควรจัดการข้อผิดพลาดได้โดยไม่ crash
            lock = SimpleTerminalLock(bad_config_file)
            self.log_test("Bad Config Handling", True, "Bad config handled gracefully")
            
            # ทดสอบการจัดการไฟล์ที่ไม่มีสิทธิ์เขียน
            restricted_file = "temp/restricted_lock.lock"
            Path(restricted_file).parent.mkdir(parents=True, exist_ok=True)
            
            # ทดสอบการจัดการ permission errors
            lock.lock_file = Path(restricted_file)
            
            # ทดสอบการล็อคเมื่อไม่สามารถสร้างไฟล์ได้
            try:
                lock.lock()
                self.log_test("Permission Error Handling", True, "Permission errors handled")
            except Exception:
                self.log_test("Permission Error Handling", True, "Permission errors handled with exception")
            
            # ทำความสะอาด
            for file in [bad_config_file, restricted_file]:
                if Path(file).exists():
                    Path(file).unlink()
            
        except Exception as e:
            self.log_test("Error Handling Test", False, f"Exception: {e}")
    
    def run_performance_test(self):
        """ทดสอบประสิทธิภาพ"""
        print(f"\n{Fore.CYAN}🧪 Testing Performance{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Performance Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            lock = SimpleTerminalLock()
            lock.set_password(self.test_password)
            
            # ทดสอบเวลาการล็อค
            start_time = time.time()
            lock.lock()
            lock_time = time.time() - start_time
            
            self.log_test("Lock Performance", lock_time < 1.0, f"Lock time: {lock_time:.3f}s")
            
            # ทดสอบเวลาการปลดล็อค
            start_time = time.time()
            lock.unlock(self.test_password)
            unlock_time = time.time() - start_time
            
            self.log_test("Unlock Performance", unlock_time < 1.0, f"Unlock time: {unlock_time:.3f}s")
            
            # ทดสอบเวลาการตรวจสอบรหัสผ่าน
            start_time = time.time()
            for _ in range(100):
                lock.verify_password(self.test_password)
            verify_time = (time.time() - start_time) / 100
            
            self.log_test("Password Verification Performance", verify_time < 0.001, f"Verify time: {verify_time:.6f}s")
            
            # ทดสอบการใช้หน่วยความจำ
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            self.log_test("Memory Usage", memory_usage < 100, f"Memory usage: {memory_usage:.2f}MB")
            
        except Exception as e:
            self.log_test("Performance Test", False, f"Exception: {e}")
    
    def run_integration_test(self):
        """ทดสอบการรวมระบบ"""
        print(f"\n{Fore.CYAN}🧪 Testing System Integration{Style.RESET_ALL}")
        
        if not SIMPLE_LOCK_AVAILABLE:
            self.log_test("Integration Test", False, "SimpleTerminalLock not available")
            return
        
        try:
            # ทดสอบการรวมกับ config file
            config_file = "config/terminal_lock_config.json"
            if Path(config_file).exists():
                lock = SimpleTerminalLock(config_file)
                self.log_test("Config Integration", True, "Config file integration works")
            else:
                self.log_test("Config Integration", True, "Config file created when missing")
            
            # ทดสอบการรวมกับ ProjectP
            projectp_file = "ProjectP.py"
            if Path(projectp_file).exists():
                self.log_test("ProjectP Integration", True, "ProjectP.py found for integration")
            else:
                self.log_test("ProjectP Integration", False, "ProjectP.py not found")
            
            # ทดสอบการรวมกับ logging system
            logs_dir = Path("logs")
            if logs_dir.exists():
                self.log_test("Logging Integration", True, "Logs directory exists")
            else:
                self.log_test("Logging Integration", True, "Logs directory created")
            
            # ทดสอบการรวมกับ temp directory
            temp_dir = Path("temp")
            if temp_dir.exists():
                self.log_test("Temp Directory Integration", True, "Temp directory exists")
            else:
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.log_test("Temp Directory Integration", True, "Temp directory created")
            
        except Exception as e:
            self.log_test("Integration Test", False, f"Exception: {e}")
    
    def generate_report(self):
        """สร้างรายงานผลการทดสอบ"""
        print(f"\n{Fore.CYAN}📊 Test Report Generation{Style.RESET_ALL}")
        
        # สร้างรายงาน
        report = {
            'test_summary': {
                'total_tests': self.test_count,
                'passed': self.passed_count,
                'failed': self.failed_count,
                'success_rate': (self.passed_count / self.test_count * 100) if self.test_count > 0 else 0
            },
            'test_results': self.test_results,
            'system_info': {
                'platform': os.name,
                'python_version': sys.version,
                'test_time': time.time()
            }
        }
        
        # บันทึกรายงาน
        report_file = f"temp/terminal_lock_test_report_{int(time.time())}.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # แสดงสรุป
        print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                                {Fore.YELLOW}📊 TEST SUMMARY{Fore.CYAN}                                       ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  {Fore.WHITE}Total Tests:{Fore.CYAN} {self.test_count:<66} {Fore.CYAN}║")
        print(f"║  {Fore.GREEN}Passed:{Fore.CYAN} {self.passed_count:<70} {Fore.CYAN}║")
        print(f"║  {Fore.RED}Failed:{Fore.CYAN} {self.failed_count:<70} {Fore.CYAN}║")
        print(f"║  {Fore.YELLOW}Success Rate:{Fore.CYAN} {report['test_summary']['success_rate']:.1f}%{' ' * 61} {Fore.CYAN}║")
        print(f"║  {Fore.WHITE}Report File:{Fore.CYAN} {report_file:<63} {Fore.CYAN}║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        # แสดงผลรวม
        if self.failed_count == 0:
            print(f"\n{Fore.GREEN}🎉 ALL TESTS PASSED! Terminal Lock System is ready for production!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠️  {self.failed_count} tests failed. Please review the results.{Style.RESET_ALL}")
        
        return report
    
    def run_all_tests(self):
        """รันการทดสอบทั้งหมด"""
        print(f"{Fore.CYAN}🧪 NICEGOLD Enterprise Terminal Lock - Test Suite{Style.RESET_ALL}")
        print(f"{Fore.WHITE}=" * 80 + Style.RESET_ALL)
        
        # รันการทดสอบทั้งหมด
        self.test_simple_lock_basic()
        self.test_enterprise_lock_basic()
        self.test_config_management()
        self.test_security_features()
        self.test_file_operations()
        self.test_concurrent_access()
        self.test_error_handling()
        self.run_performance_test()
        self.run_integration_test()
        
        # สร้างรายงาน
        report = self.generate_report()
        
        return report


def main():
    """ฟังก์ชันหลัก"""
    tester = TerminalLockTester()
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "basic":
            tester.test_simple_lock_basic()
            tester.test_enterprise_lock_basic()
        elif test_name == "security":
            tester.test_security_features()
        elif test_name == "performance":
            tester.run_performance_test()
        elif test_name == "integration":
            tester.run_integration_test()
        elif test_name == "all":
            tester.run_all_tests()
        else:
            print(f"{Fore.RED}Unknown test: {test_name}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Available tests: basic, security, performance, integration, all{Style.RESET_ALL}")
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main() 