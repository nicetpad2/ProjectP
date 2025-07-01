#!/usr/bin/env python3
"""
🌟 NICEGOLD ENTERPRISE MENU 1 LOGGER SYSTEM
ระบบ Logging สำหรับ Menu 1 Full Pipeline โดยเฉพาะ
Advanced Progress Tracking & Real-time Monitoring for Menu 1

🎯 Enterprise Features:
- สีสันสวยงามและถนอมสายตา
- ติดตาม Progress แบบ Real-time
- การจัดการ Error/Warning ที่ครบถ้วน
- ระบบ Reporting ที่สมบูรณ์
- บันทึกการทำงานแบบ Enterprise Grade
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Callable
import json
import threading
from enum import Enum
from collections import defaultdict
import colorama
from colorama import Fore, Style, Back

# Initialize colorama for Windows
colorama.init(autoreset=True)


class ProcessStatus(Enum):
    """สถานะการประมวลผล"""
    STARTING = "🚀 STARTING"
    RUNNING = "⚡ RUNNING"
    SUCCESS = "✅ SUCCESS"
    WARNING = "⚠️ WARNING"
    ERROR = "❌ ERROR"
    CRITICAL = "🔥 CRITICAL"
    COMPLETED = "🎉 COMPLETED"


class Menu1Logger:
    """
    Enterprise Logger System สำหรับ Menu 1 Full Pipeline
    ระบบ Logging ที่สมบูรณ์แบบสำหรับ Menu 1 โดยเฉพาะ
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"menu1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.processes = {}
        self.step_counter = 0
        self.error_count = 0
        self.warning_count = 0
        self.success_count = 0
        self.lock = threading.Lock()
        
        # สร้าง log directories
        self._create_log_directories()
        
        # Setup logging
        self._setup_enterprise_logging()
        
        # Track performance metrics
        self.performance_metrics = {
            'total_steps': 0,
            'completed_steps': 0,
            'errors': [],
            'warnings': [],
            'processing_times': {}
        }
        
        # Color scheme สำหรับ Menu 1
        self.colors = {
            'header': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'error': Fore.RED + Style.BRIGHT,
            'critical': Fore.MAGENTA + Style.BRIGHT,
            'info': Fore.BLUE + Style.BRIGHT,
            'progress': Fore.CYAN,
            'step': Fore.WHITE + Style.BRIGHT,
            'reset': Style.RESET_ALL
        }
    
    def _create_log_directories(self):
        """สร้าง directory structure สำหรับ logs"""
        directories = [
            'logs',
            'logs/menu1',
            'logs/menu1/processes',
            'logs/menu1/errors',
            'logs/menu1/warnings',
            'logs/menu1/performance',
            'logs/menu1/sessions'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_enterprise_logging(self):
        """Setup enterprise-grade logging system"""
        # Create logger
        self.logger = logging.getLogger(f'Menu1_{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with beautiful formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for session logs
        session_file = f'logs/menu1/sessions/{self.session_id}.log'
        file_handler = logging.FileHandler(session_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_file = f'logs/menu1/errors/{self.session_id}_errors.log'
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def start_menu1_session(self):
        """เริ่มต้น Menu 1 Session"""
        header = f"""
{self.colors['header']}{'='*80}
🌟 NICEGOLD ENTERPRISE - MENU 1 FULL PIPELINE SESSION STARTED
{'='*80}{self.colors['reset']}

{self.colors['info']}📅 Session ID: {self.session_id}
🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
🎯 Target: AUC ≥ 70% | 🛡️ Enterprise Grade | 📊 Real Data Only{self.colors['reset']}

{self.colors['header']}{'='*80}{self.colors['reset']}
"""
        print(header)
        self.logger.info(f"Menu 1 Full Pipeline Session Started - {self.session_id}")
    
    def log_step(self, step_num: int, step_name: str, status: ProcessStatus, 
                 details: str = "", progress: int = 0):
        """
        บันทึก Step ต่างๆ ของ Menu 1 Pipeline
        """
        with self.lock:
            self.step_counter += 1
            timestamp = datetime.now()
            
            # สร้าง progress bar
            progress_bar = self._create_progress_bar(progress)
            
            # กำหนดสี status
            if status == ProcessStatus.SUCCESS:
                status_color = self.colors['success']
                self.success_count += 1
            elif status == ProcessStatus.WARNING:
                status_color = self.colors['warning']
                self.warning_count += 1
            elif status in [ProcessStatus.ERROR, ProcessStatus.CRITICAL]:
                status_color = self.colors['error']
                self.error_count += 1
            else:
                status_color = self.colors['info']
            
            # Format message
            message = f"""
{self.colors['step']}┌─ STEP {step_num:02d}: {step_name} ─{self.colors['reset']}
{status_color}├─ Status: {status.value}{self.colors['reset']}
{self.colors['progress']}├─ Progress: {progress_bar} ({progress}%){self.colors['reset']}"""
            
            if details:
                message += f"\n{self.colors['info']}├─ Details: {details}{self.colors['reset']}"
            
            message += f"\n{self.colors['step']}└─ Time: {timestamp.strftime('%H:%M:%S')}{self.colors['reset']}\n"
            
            print(message)
            
            # บันทึกใน log file
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'step_number': step_num,
                'step_name': step_name,
                'status': status.value,
                'details': details,
                'progress': progress
            }
            
            self.processes[f'step_{step_num}'] = log_entry
            self.logger.info(f"Step {step_num}: {step_name} - {status.value} - {details}")
    
    def _create_progress_bar(self, progress: int, width: int = 30) -> str:
        """สร้าง progress bar ที่สวยงาม"""
        filled = int(width * progress / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def log_error(self, error_msg: str, exception: Exception = None, 
                  step_name: str = "Unknown"):
        """
        บันทึก Error แบบ Enterprise Grade
        """
        with self.lock:
            self.error_count += 1
            timestamp = datetime.now()
            
            error_details = {
                'timestamp': timestamp.isoformat(),
                'step_name': step_name,
                'error_message': error_msg,
                'exception_type': type(exception).__name__ if exception else None,
                'exception_details': str(exception) if exception else None,
                'traceback': None
            }
            
            if exception:
                import traceback as tb
                error_details['traceback'] = tb.format_exc()
            
            # แสดงข้อผิดพลาดในรูปแบบที่สวยงาม
            error_display = f"""
{self.colors['error']}{'='*60}
🔥 ERROR DETECTED - IMMEDIATE ATTENTION REQUIRED
{'='*60}{self.colors['reset']}

{self.colors['error']}📍 Step: {step_name}
💥 Error: {error_msg}
🕐 Time: {timestamp.strftime('%H:%M:%S')}
{self.colors['reset']}"""
            
            if exception:
                error_display += f"\n{self.colors['error']}🔧 Exception: {type(exception).__name__}: {str(exception)}{self.colors['reset']}"
            
            error_display += f"\n{self.colors['error']}{'='*60}{self.colors['reset']}\n"
            
            print(error_display)
            
            # บันทึกใน performance metrics
            self.performance_metrics['errors'].append(error_details)
            
            # บันทึกใน log file
            self.logger.error(f"ERROR in {step_name}: {error_msg}")
            if exception:
                self.logger.error(f"Exception: {type(exception).__name__}: {str(exception)}")
    
    def log_warning(self, warning_msg: str, step_name: str = "Unknown"):
        """
        บันทึก Warning แบบ Enterprise Grade
        """
        with self.lock:
            self.warning_count += 1
            timestamp = datetime.now()
            
            warning_details = {
                'timestamp': timestamp.isoformat(),
                'step_name': step_name,
                'warning_message': warning_msg
            }
            
            # แสดง warning ในรูปแบบที่สวยงาม
            warning_display = f"""
{self.colors['warning']}⚠️ WARNING: {warning_msg}
📍 Step: {step_name} | 🕐 Time: {timestamp.strftime('%H:%M:%S')}{self.colors['reset']}
"""
            
            print(warning_display)
            
            # บันทึกใน performance metrics
            self.performance_metrics['warnings'].append(warning_details)
            
            # บันทึกใน log file
            self.logger.warning(f"WARNING in {step_name}: {warning_msg}")
    
    def log_success(self, success_msg: str, step_name: str = "Unknown", 
                    metrics: Dict = None):
        """
        บันทึก Success แบบ Enterprise Grade
        """
        with self.lock:
            self.success_count += 1
            timestamp = datetime.now()
            
            # แสดง success ในรูปแบบที่สวยงาม
            success_display = f"""
{self.colors['success']}✅ SUCCESS: {success_msg}
📍 Step: {step_name} | 🕐 Time: {timestamp.strftime('%H:%M:%S')}{self.colors['reset']}"""
            
            if metrics:
                success_display += f"\n{self.colors['info']}📊 Metrics: {metrics}{self.colors['reset']}"
            
            success_display += "\n"
            
            print(success_display)
            
            # บันทึกใน log file
            self.logger.info(f"SUCCESS in {step_name}: {success_msg}")
            if metrics:
                self.logger.info(f"Metrics: {metrics}")
    
    def complete_menu1_session(self, final_results: Dict):
        """
        จบ Menu 1 Session และสร้าง comprehensive report
        """
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # คำนวณ success rate
        total_operations = self.success_count + self.error_count + self.warning_count
        success_rate = (self.success_count / total_operations * 100) if total_operations > 0 else 0
        
        # สร้าง final report
        report = f"""
{self.colors['header']}{'='*80}
🎉 MENU 1 FULL PIPELINE SESSION COMPLETED
{'='*80}{self.colors['reset']}

{self.colors['info']}📅 Session ID: {self.session_id}
🕐 Duration: {duration}
📊 Success Rate: {success_rate:.1f}%

📈 OPERATION SUMMARY:
   ✅ Successes: {self.success_count}
   ⚠️ Warnings: {self.warning_count}
   ❌ Errors: {self.error_count}

🎯 FINAL RESULTS:{self.colors['reset']}"""
        
        # แสดงผลลัพธ์สุดท้าย
        for key, value in final_results.items():
            if isinstance(value, float):
                report += f"\n{self.colors['success']}   {key}: {value:.4f}{self.colors['reset']}"
            else:
                report += f"\n{self.colors['success']}   {key}: {value}{self.colors['reset']}"
        
        # การประเมินคุณภาพ
        if success_rate >= 90:
            quality_grade = f"{self.colors['success']}🏆 EXCELLENT{self.colors['reset']}"
        elif success_rate >= 80:
            quality_grade = f"{self.colors['success']}✅ GOOD{self.colors['reset']}"
        elif success_rate >= 70:
            quality_grade = f"{self.colors['warning']}⚠️ ACCEPTABLE{self.colors['reset']}"
        else:
            quality_grade = f"{self.colors['error']}❌ NEEDS IMPROVEMENT{self.colors['reset']}"
        
        report += f"\n\n{self.colors['info']}🏅 Quality Grade: {quality_grade}"
        report += f"\n\n{self.colors['header']}{'='*80}{self.colors['reset']}\n"
        
        print(report)
        
        # บันทึก comprehensive report
        self._save_comprehensive_report(final_results, duration, success_rate)
        
        # บันทึกใน log file
        self.logger.info(f"Menu 1 Session Completed - Duration: {duration}, Success Rate: {success_rate:.1f}%")
        self.logger.info(f"Final Results: {final_results}")
    
    def _save_comprehensive_report(self, final_results: Dict, duration, success_rate: float):
        """
        บันทึก comprehensive report ในรูปแบบ JSON
        """
        report_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration)
            },
            'operation_summary': {
                'total_steps': self.step_counter,
                'success_count': self.success_count,
                'warning_count': self.warning_count,
                'error_count': self.error_count,
                'success_rate': success_rate
            },
            'final_results': final_results,
            'performance_metrics': self.performance_metrics,
            'all_processes': self.processes
        }
        
        # บันทึกเป็น JSON file
        report_file = f'logs/menu1/sessions/{self.session_id}_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"{self.colors['info']}📋 Comprehensive report saved: {report_file}{self.colors['reset']}")
    
    def get_session_stats(self) -> Dict:
        """
        ดึง session statistics
        """
        return {
            'session_id': self.session_id,
            'total_steps': self.step_counter,
            'success_count': self.success_count,
            'warning_count': self.warning_count,
            'error_count': self.error_count,
            'duration': str(datetime.now() - self.start_time)
        }


# สร้าง singleton instance สำหรับ Menu 1
_menu1_logger = None

def get_menu1_logger(session_id: str = None) -> Menu1Logger:
    """
    ดึง Menu1Logger instance (Singleton pattern)
    """
    global _menu1_logger
    if _menu1_logger is None or session_id:
        _menu1_logger = Menu1Logger(session_id)
    return _menu1_logger


# Convenience functions สำหรับใช้งานง่าย
def log_step(step_num: int, step_name: str, status: ProcessStatus, 
             details: str = "", progress: int = 0):
    """บันทึก step ใน Menu 1 Pipeline"""
    logger = get_menu1_logger()
    logger.log_step(step_num, step_name, status, details, progress)


def log_error(error_msg: str, exception: Exception = None, step_name: str = "Unknown"):
    """บันทึก error ใน Menu 1 Pipeline"""
    logger = get_menu1_logger()
    logger.log_error(error_msg, exception, step_name)


def log_warning(warning_msg: str, step_name: str = "Unknown"):
    """บันทึก warning ใน Menu 1 Pipeline"""
    logger = get_menu1_logger()
    logger.log_warning(warning_msg, step_name)


def log_success(success_msg: str, step_name: str = "Unknown", metrics: Dict = None):
    """บันทึก success ใน Menu 1 Pipeline"""
    logger = get_menu1_logger()
    logger.log_success(success_msg, step_name, metrics)


def start_menu1_session(session_id: str = None):
    """เริ่ม Menu 1 Session"""
    logger = get_menu1_logger(session_id)
    logger.start_menu1_session()
    return logger


def complete_menu1_session(final_results: Dict):
    """จบ Menu 1 Session"""
    logger = get_menu1_logger()
    logger.complete_menu1_session(final_results)
