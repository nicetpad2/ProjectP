#!/usr/bin/env python3
"""
📊 NICEGOLD ENTERPRISE LOGGER
ระบบการบันทึกข้อมูลระดับ Enterprise
Advanced Logging System with Progress Tracking & Real-time Monitoring

🎯 Features:
- Beautiful colored console output
- Progress tracking with visual bars
- Real-time error/warning alerts
- Comprehensive file logging
- Performance monitoring
- Process status tracking
"""

import logging
import sys
import platform
from datetime import datetime
from typing import Dict, List, Callable
import os
import io
import json
import traceback
import threading
from pathlib import Path
from enum import Enum
from collections import defaultdict
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows
colorama.init(autoreset=True)


class LogLevel(Enum):
    """Log Levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"
    PROGRESS = "PROGRESS"


class ProcessStatus(Enum):
    """Process Status"""
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING = "WAITING"


class ProgressTracker:
    """Advanced Progress Tracking System"""
    
    def __init__(self):
        self.processes = {}
        self.alerts = []
        self.lock = threading.Lock()
        self.error_count = defaultdict(int)
        self.warning_count = defaultdict(int)
    
    def start_process(self, process_id: str, name: str, total_steps: int = 0):
        """Start tracking a process"""
        with self.lock:
            self.processes[process_id] = {
                'name': name,
                'status': ProcessStatus.STARTED,
                'start_time': datetime.now(),
                'current_step': 0,
                'total_steps': total_steps,
                'messages': [],
                'errors': [],
                'warnings': []
            }
    
    def update_progress(self, process_id: str, step: int, message: str = ""):
        """Update process progress"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['current_step'] = step
                self.processes[process_id]['status'] = ProcessStatus.IN_PROGRESS
                if message:
                    self.processes[process_id]['messages'].append({
                        'timestamp': datetime.now(),
                        'message': message
                    })
    
    def complete_process(self, process_id: str, success: bool = True):
        """Mark process as completed"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['status'] = ProcessStatus.COMPLETED if success else ProcessStatus.FAILED
                self.processes[process_id]['end_time'] = datetime.now()
    
    def add_error(self, process_id: str, error: str):
        """Add error to process"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['errors'].append({
                    'timestamp': datetime.now(),
                    'error': error
                })
                self.error_count[process_id] += 1
    
    def add_warning(self, process_id: str, warning: str):
        """Add warning to process"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['warnings'].append({
                    'timestamp': datetime.now(),
                    'warning': warning
                })
                self.warning_count[process_id] += 1
    
    def get_status(self, process_id: str) -> Dict:
        """Get process status"""
        with self.lock:
            return self.processes.get(process_id, {})
    
    def get_progress_display(self, process_id: str) -> str:
        """Get formatted progress display"""
        process = self.get_status(process_id)
        if not process:
            return "Process not found"
        
        current = process.get('current_step', 0)
        total = process.get('total_steps', 0)
        status = process.get('status', ProcessStatus.WAITING)
        
        if total > 0:
            percentage = (current / total) * 100
            progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            return f"[{progress_bar}] {current}/{total} ({percentage:.1f}%) - {status.value}"
        else:
            return f"Step {current} - {status.value}"


class EnterpriseLogger:
    """Enterprise Logger Class with Advanced Features"""
    
    def __init__(self, name: str = "NICEGOLD", 
                 log_level: str = "INFO",
                 enable_colors: bool = True,
                 enable_file_logging: bool = True):
        self.name = name
        self.enable_colors = enable_colors
        self.enable_file_logging = enable_file_logging
        self.logger = logging.getLogger(name)
        self.progress_tracker = ProgressTracker()
        self.alert_callbacks: List[Callable] = []
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/errors", exist_ok=True)
        os.makedirs("logs/warnings", exist_ok=True)
        
        self.setup_logger(log_level)
        
        # Performance metrics
        self.start_time = datetime.now()
        self.log_counts = defaultdict(int)
    
    def setup_logger(self, log_level: str = "INFO"):
        """Setup advanced logger with multiple handlers - improved error handling"""
        # Clear existing handlers safely
        for handler in self.logger.handlers[:]:
            try:
                handler.close()
            except:
                pass
            self.logger.removeHandler(handler)
        
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler with colors
        try:
            console_handler = self._create_console_handler()
            self.logger.addHandler(console_handler)
        except Exception as e:
            print(f"⚠️ Could not create console handler: {e}")
        
        if self.enable_file_logging:
            try:
                # Main log file handler
                main_file_handler = self._create_file_handler("main")
                self.logger.addHandler(main_file_handler)
                
                # Error-specific handler
                error_handler = self._create_file_handler("error", logging.ERROR)
                self.logger.addHandler(error_handler)
                
                # Warning-specific handler
                warning_handler = self._create_file_handler("warning", logging.WARNING)
                self.logger.addHandler(warning_handler)
            except Exception as e:
                print(f"⚠️ Could not create file handlers: {e}")
                # Continue without file logging if it fails
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create colored console handler - with improved error handling"""
        try:
            # Try to create a proper UTF-8 handler
            handler = logging.StreamHandler(
                io.TextIOWrapper(
                    sys.stdout.buffer, 
                    encoding='utf-8', 
                    errors='replace'
                )
            )
        except (AttributeError, OSError):
            try:
                # Fallback to standard stdout
                handler = logging.StreamHandler(sys.stdout)
            except:
                # Last resort: stderr
                handler = logging.StreamHandler(sys.stderr)
        
        # Add safety filter to prevent closed stream errors
        def safe_console_filter(record):
            try:
                # Test if the stream is still open
                handler.stream.write('')
                return True
            except (ValueError, OSError):
                # Stream is closed, skip this record
                return False
        
        handler.addFilter(safe_console_filter)
        
        if self.enable_colors:
            formatter = ColoredFormatter(
                '%(asctime)s | %(levelname_color)s%(levelname)s%(reset)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        return handler
    
    def _create_file_handler(self, file_type: str, min_level: int = logging.DEBUG) -> logging.FileHandler:
        """Create file handler for specific log types - with improved error handling"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        if file_type == "error":
            filename = f"logs/errors/errors_{timestamp}.log"
        elif file_type == "warning":
            filename = f"logs/warnings/warnings_{timestamp}.log"
        else:
            filename = f"logs/nicegold_enterprise_{timestamp}.log"
        
        try:
            # Create handler with explicit error handling
            handler = logging.FileHandler(filename, encoding='utf-8', mode='a')
            
            # Add a custom filter to prevent closed file errors
            def safe_filter(record):
                try:
                    return True
                except (ValueError, OSError):
                    return False
            
            handler.addFilter(safe_filter)
            
        except (OSError, UnicodeError) as e:
            print(f"⚠️ Warning: Could not create file handler for {filename}: {e}")
            # Return a null handler if file creation fails
            handler = logging.NullHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        handler.setLevel(min_level)
        return handler
    
    def start_process_tracking(self, process_id: str, name: str, total_steps: int = 0):
        """Start tracking a process"""
        self.progress_tracker.start_process(process_id, name, total_steps)
        self.info(f"🚀 Started process: {name} (ID: {process_id})")
    
    def update_process_progress(self, process_id: str, step: int, message: str = ""):
        """Update process progress"""
        self.progress_tracker.update_progress(process_id, step, message)
        progress_display = self.progress_tracker.get_progress_display(process_id)
        self.info(f"📊 Progress Update: {progress_display}")
        if message:
            self.info(f"   └─ {message}")
    
    def complete_process(self, process_id: str, success: bool = True):
        """Complete process tracking"""
        self.progress_tracker.complete_process(process_id, success)
        if success:
            self.success(f"✅ Process completed successfully: {process_id}")
        else:
            self.error(f"❌ Process failed: {process_id}")
    
    def info(self, message: str, process_id: str = None):
        """Log info message"""
        safe_message = safe_log_message(message)
        self.logger.info(safe_message)
        self.log_counts['info'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, message)
    
    def success(self, message: str, process_id: str = None):
        """Log success message"""
        safe_message = safe_log_message(f"✅ {message}")
        self.logger.info(safe_message)
        self.log_counts['success'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, message)
    
    def warning(self, message: str, process_id: str = None):
        """Log warning message"""
        safe_message = safe_log_message(f"⚠️ WARNING: {message}")
        self.logger.warning(safe_message)
        self.log_counts['warning'] += 1
        
        if process_id:
            self.progress_tracker.add_warning(process_id, message)
        
        self._trigger_alert("WARNING", message)
    
    def error(self, message: str, process_id: str = None, exception: Exception = None):
        """Log error message with optional exception details"""
        error_msg = f"❌ ERROR: {message}"
        
        if exception:
            error_msg += f"\n   Exception: {str(exception)}"
            error_msg += f"\n   Traceback: {traceback.format_exc()}"
        
        safe_message = safe_log_message(error_msg)
        self.logger.error(safe_message)
        self.log_counts['error'] += 1
        
        if process_id:
            self.progress_tracker.add_error(process_id, message)
        
        self._trigger_alert("ERROR", message)
        
        # Save detailed error report
        self._save_error_report(message, exception)
    
    def critical(self, message: str, process_id: str = None, exception: Exception = None):
        """Log critical message"""
        critical_msg = f"🚨 CRITICAL: {message}"
        
        if exception:
            critical_msg += f"\n   Exception: {str(exception)}"
            critical_msg += f"\n   Traceback: {traceback.format_exc()}"
        
        safe_message = safe_log_message(critical_msg)
        self.logger.critical(safe_message)
        self.log_counts['critical'] += 1
        
        if process_id:
            self.progress_tracker.add_error(process_id, f"CRITICAL: {message}")
        
        self._trigger_alert("CRITICAL", message)
        
        # Save critical error report
        self._save_error_report(message, exception, critical=True)
    
    def debug(self, message: str, process_id: str = None):
        """Log debug message"""
        safe_message = safe_log_message(f"🔍 DEBUG: {message}")
        self.logger.debug(safe_message)
        self.log_counts['debug'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, f"DEBUG: {message}")
    
    def _trigger_alert(self, level: str, message: str):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(level, message)
            except Exception as e:
                print(f"Alert callback failed: {e}")
    
    def _save_error_report(self, message: str, exception: Exception = None, critical: bool = False):
        """Save detailed error report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_type = "critical" if critical else "error"
        filename = f"logs/errors/{report_type}_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'level': 'CRITICAL' if critical else 'ERROR',
            'message': message,
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'system_info': {
                'platform': platform.system(),
                'python_version': sys.version,
                'logger_name': self.name
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Failed to save error report: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        runtime = datetime.now() - self.start_time
        return {
            'runtime_seconds': runtime.total_seconds(),
            'log_counts': dict(self.log_counts),
            'total_logs': sum(self.log_counts.values()),
            'error_rate': self.log_counts['error'] / max(sum(self.log_counts.values()), 1) * 100,
            'warning_rate': self.log_counts['warning'] / max(sum(self.log_counts.values()), 1) * 100
        }
    
    def display_performance_summary(self):
        """Display performance summary"""
        summary = self.get_performance_summary()
        print("\n" + "="*60)
        print("📊 LOGGING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
        print(f"Total Logs: {summary['total_logs']}")
        print(f"Error Rate: {summary['error_rate']:.2f}%")
        print(f"Warning Rate: {summary['warning_rate']:.2f}%")
        print("\nLog Breakdown:")
        for level, count in summary['log_counts'].items():
            print(f"  {level.upper()}: {count}")
        print("="*60)
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA,
        'SUCCESS': Fore.GREEN + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color information to record
        record.levelname_color = self.COLORS.get(record.levelname, '')
        record.reset = Style.RESET_ALL
        return super().format(record)


def setup_enterprise_logger(log_level: str = "INFO") -> logging.Logger:
    """Legacy function for backward compatibility"""
    logger = EnterpriseLogger("NICEGOLD_Enterprise", log_level)
    return logger.logger


def safe_log_message(message: str) -> str:
    """สร้างข้อความ log ที่ปลอดภัยสำหรับทุกระบบ"""
    # Replace emoji with safe text representations
    emoji_replacements = {
        '🚀': '[ROCKET]',
        '✅': '[CHECK]',
        '❌': '[X]',
        '⚠️': '[WARNING]',
        'ℹ️': '[INFO]',
        '🔍': '[SEARCH]',
        '💥': '[EXPLOSION]',
        '🛑': '[STOP]',
        '🌊': '[WAVE]',
        '📊': '[CHART]',
        '🎯': '[TARGET]',
        '🧠': '[BRAIN]',
        '🤖': '[ROBOT]',
        '🏆': '[TROPHY]',
        '⚡': '[LIGHTNING]',
        '🔗': '[LINK]',
        '📈': '[CHART_UP]',
        '🎉': '[PARTY]',
        '📁': '[FOLDER]',
        '🎛️': '[CONTROL]',
        '🏢': '[BUILDING]',
        '🚨': '[ALERT]',
        '📋': '[CLIPBOARD]',
        '🔧': '[WRENCH]',
        '💡': '[BULB]',
        '🎨': '[PALETTE]',
        '🔥': '[FIRE]',
        '💎': '[DIAMOND]',
        '🌟': '[STAR]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    return safe_message


# Global logger instance
_global_logger = None


def get_enterprise_logger(name: str = "NICEGOLD") -> EnterpriseLogger:
    """Get global enterprise logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EnterpriseLogger(name)
    return _global_logger


class ProcessManager:
    """Advanced Process Management for Menu Systems"""
    
    def __init__(self, logger: EnterpriseLogger):
        self.logger = logger
        self.active_processes = {}
        self.process_history = []
    
    def start_menu_process(self, menu_id: str, menu_name: str, 
                          steps: List[str]) -> str:
        """Start a menu process with defined steps"""
        process_id = f"menu_{menu_id}_{datetime.now().strftime('%H%M%S')}"
        
        self.logger.start_process_tracking(
            process_id, 
            f"Menu {menu_id}: {menu_name}", 
            len(steps)
        )
        
        self.active_processes[process_id] = {
            'menu_id': menu_id,
            'menu_name': menu_name,
            'steps': steps,
            'current_step': 0,
            'start_time': datetime.now(),
            'status': 'RUNNING'
        }
        
        return process_id
    
    def execute_step(self, process_id: str, step_function: Callable, 
                    step_name: str, *args, **kwargs) -> bool:
        """Execute a step with error handling and tracking"""
        if process_id not in self.active_processes:
            self.logger.error(f"Process {process_id} not found")
            return False
        
        process = self.active_processes[process_id]
        current_step = process['current_step']
        
        self.logger.info(
            f"🔄 Executing Step {current_step + 1}: {step_name}",
            process_id
        )
        
        try:
            # Execute the step function
            result = step_function(*args, **kwargs)
            
            # Update progress
            process['current_step'] += 1
            self.logger.update_process_progress(
                process_id,
                process['current_step'],
                f"Completed: {step_name}"
            )
            
            self.logger.success(f"✅ Step completed: {step_name}", process_id)
            return True
            
        except Exception as e:
            self.logger.error(
                f"Step failed: {step_name}",
                process_id,
                e
            )
            process['status'] = 'FAILED'
            self.logger.complete_process(process_id, False)
            return False
    
    def complete_menu_process(self, process_id: str, success: bool = True):
        """Complete a menu process"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            process['end_time'] = datetime.now()
            process['status'] = 'COMPLETED' if success else 'FAILED'
            
            # Move to history
            self.process_history.append(process)
            del self.active_processes[process_id]
            
            self.logger.complete_process(process_id, success)
    
    def get_process_summary(self, process_id: str) -> Dict:
        """Get detailed process summary"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            return {
                'process_id': process_id,
                'menu_name': process['menu_name'],
                'status': process['status'],
                'progress': f"{process['current_step']}/{len(process['steps'])}",
                'current_step_name': process['steps'][process['current_step']] if process['current_step'] < len(process['steps']) else "Completed",
                'runtime': (datetime.now() - process['start_time']).total_seconds()
            }
        return {}


class ErrorReporter:
    """Advanced Error Reporting System"""
    
    def __init__(self, logger: EnterpriseLogger):
        self.logger = logger
        self.error_history = []
        self.error_patterns = defaultdict(int)
    
    def report_error(self, error_type: str, message: str, 
                    context: Dict = None, critical: bool = False):
        """Report error with context"""
        error_report = {
            'timestamp': datetime.now(),
            'type': error_type,
            'message': message,
            'context': context or {},
            'critical': critical
        }
        
        self.error_history.append(error_report)
        self.error_patterns[error_type] += 1
        
        if critical:
            self.logger.critical(f"{error_type}: {message}")
        else:
            self.logger.error(f"{error_type}: {message}")
        
        # Auto-suggest solutions for common errors
        suggestion = self._get_error_suggestion(error_type, message)
        if suggestion:
            self.logger.info(f"💡 Suggestion: {suggestion}")
    
    def _get_error_suggestion(self, error_type: str, message: str) -> str:
        """Get error suggestion based on pattern"""
        suggestions = {
            'FileNotFoundError': "Check if the file path exists and is accessible",
            'ImportError': "Verify that all required packages are installed",
            'MemoryError': "Consider reducing data size or increasing system memory",
            'TimeoutError': "Check network connectivity or increase timeout value",
            'ValueError': "Validate input data format and ranges",
            'KeyError': "Verify that all required keys exist in the data structure"
        }
        
        for pattern, suggestion in suggestions.items():
            if pattern.lower() in error_type.lower() or pattern.lower() in message.lower():
                return suggestion
        
        return ""
    
    def get_error_summary(self) -> Dict:
        """Get error summary and patterns"""
        return {
            'total_errors': len(self.error_history),
            'error_patterns': dict(self.error_patterns),
            'recent_errors': self.error_history[-5:] if self.error_history else [],
            'critical_errors': [e for e in self.error_history if e.get('critical', False)]
        }


# Export main classes and functions
__all__ = [
    'EnterpriseLogger',
    'ProcessManager', 
    'ErrorReporter',
    'ProcessStatus',
    'LogLevel',
    'get_enterprise_logger',
    'setup_enterprise_logger',
    'safe_log_message'
]
