#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE ADVANCED LOGGER
ระบบ Logging ขั้นสูงพร้อม Progress Tracking & Real-time Monitoring
"""

import logging
import sys
import platform
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
import os
import io
import json
import traceback
import threading
from enum import Enum
from collections import defaultdict

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


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


class AdvancedProgressTracker:
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
                status = ProcessStatus.COMPLETED if success else ProcessStatus.FAILED
                self.processes[process_id]['status'] = status
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
            bar_length = 20
            filled_length = int(percentage / 5)
            progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)
            return f"[{progress_bar}] {current}/{total} ({percentage:.1f}%) - {status.value}"
        else:
            return f"Step {current} - {status.value}"


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = {
            'DEBUG': Fore.CYAN if COLORS_AVAILABLE else '',
            'INFO': Fore.GREEN if COLORS_AVAILABLE else '',
            'WARNING': Fore.YELLOW if COLORS_AVAILABLE else '',
            'ERROR': Fore.RED if COLORS_AVAILABLE else '',
            'CRITICAL': Fore.MAGENTA if COLORS_AVAILABLE else '',
            'SUCCESS': (Fore.GREEN + Style.BRIGHT) if COLORS_AVAILABLE else ''
        }
        self.reset = Style.RESET_ALL if COLORS_AVAILABLE else ''
    
    def format(self, record):
        # Add color information to record
        record.levelname_color = self.colors.get(record.levelname, '')
        record.reset = self.reset
        return super().format(record)


class EnterpriseAdvancedLogger:
    """Enterprise Advanced Logger with Progress Tracking"""
    
    def __init__(self, name: str = "NICEGOLD_ADVANCED", 
                 log_level: str = "INFO",
                 enable_colors: bool = True,
                 enable_file_logging: bool = True):
        self.name = name
        self.enable_colors = enable_colors and COLORS_AVAILABLE
        self.enable_file_logging = enable_file_logging
        self.logger = logging.getLogger(name)
        self.progress_tracker = AdvancedProgressTracker()
        self.alert_callbacks: List[Callable] = []
        
        # Create logs directory structure
        self._create_log_directories()
        
        # Setup logger
        self.setup_logger(log_level)
        
        # Performance metrics
        self.start_time = datetime.now()
        self.log_counts = defaultdict(int)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _create_log_directories(self):
        """Create necessary log directories"""
        directories = [
            "logs",
            "logs/errors",
            "logs/warnings", 
            "logs/processes",
            "logs/performance"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_logger(self, log_level: str = "INFO"):
        """Setup advanced logger with multiple handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler with colors
        console_handler = self._create_console_handler()
        self.logger.addHandler(console_handler)
        
        if self.enable_file_logging:
            # Main log file handler
            main_file_handler = self._create_file_handler("main")
            self.logger.addHandler(main_file_handler)
            
            # Error-specific handler
            error_handler = self._create_file_handler("error", logging.ERROR)
            self.logger.addHandler(error_handler)
            
            # Warning-specific handler
            warning_handler = self._create_file_handler("warning", logging.WARNING)
            self.logger.addHandler(warning_handler)
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create colored console handler"""
        try:
            handler = logging.StreamHandler(
                io.TextIOWrapper(
                    sys.stdout.buffer, 
                    encoding='utf-8', 
                    errors='replace'
                )
            )
        except (AttributeError, OSError):
            handler = logging.StreamHandler(sys.stdout)
        
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
        """Create file handler for specific log types"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        if file_type == "error":
            filename = f"logs/errors/errors_{timestamp}.log"
        elif file_type == "warning":
            filename = f"logs/warnings/warnings_{timestamp}.log"
        else:
            filename = f"logs/nicegold_advanced_{timestamp}.log"
        
        try:
            handler = logging.FileHandler(filename, encoding='utf-8')
        except (OSError, UnicodeError):
            handler = logging.FileHandler(filename)
        
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
        self.info(f"📊 Progress: {progress_display}")
        if message:
            self.info(f"   └─ {message}")
    
    def complete_process(self, process_id: str, success: bool = True):
        """Complete process tracking"""
        self.progress_tracker.complete_process(process_id, success)
        if success:
            self.success(f"✅ Process completed: {process_id}")
        else:
            self.error(f"❌ Process failed: {process_id}")
    
    def info(self, message: str, process_id: str = None):
        """Log info message"""
        safe_message = self._safe_log_message(message)
        self.logger.info(safe_message)
        self.log_counts['info'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, message)
    
    def success(self, message: str, process_id: str = None):
        """Log success message"""
        safe_message = self._safe_log_message(f"✅ {message}")
        self.logger.info(safe_message)
        self.log_counts['success'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, message)
    
    def warning(self, message: str, process_id: str = None):
        """Log warning message"""
        safe_message = self._safe_log_message(f"⚠️ WARNING: {message}")
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
        
        safe_message = self._safe_log_message(error_msg)
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
        
        safe_message = self._safe_log_message(critical_msg)
        self.logger.critical(safe_message)
        self.log_counts['critical'] += 1
        
        if process_id:
            self.progress_tracker.add_error(process_id, f"CRITICAL: {message}")
        
        self._trigger_alert("CRITICAL", message)
        
        # Save critical error report
        self._save_error_report(message, exception, critical=True)
    
    def debug(self, message: str, process_id: str = None):
        """Log debug message"""
        safe_message = self._safe_log_message(f"🔍 DEBUG: {message}")
        self.logger.debug(safe_message)
        self.log_counts['debug'] += 1
        
        if process_id:
            self.progress_tracker.update_progress(process_id, 0, f"DEBUG: {message}")
    
    def _safe_log_message(self, message: str) -> str:
        """Convert message to safe format"""
        emoji_replacements = {
            '🚀': '[ROCKET]', '✅': '[CHECK]', '❌': '[X]', '⚠️': '[WARNING]',
            '🔍': '[SEARCH]', '📊': '[CHART]', '🎯': '[TARGET]', '🧠': '[BRAIN]',
            '🤖': '[ROBOT]', '🏆': '[TROPHY]', '⚡': '[LIGHTNING]', '🔗': '[LINK]',
            '📈': '[CHART_UP]', '🎉': '[PARTY]', '📁': '[FOLDER]', '🎛️': '[CONTROL]',
            '🏢': '[BUILDING]', '🚨': '[ALERT]', '📋': '[CLIPBOARD]', '🔧': '[WRENCH]',
            '💡': '[BULB]', '🎨': '[PALETTE]', '🔥': '[FIRE]', '💎': '[DIAMOND]',
            '🌟': '[STAR]', '🌊': '[WAVE]', '💥': '[EXPLOSION]', '🛑': '[STOP]'
        }
        
        safe_message = message
        for emoji, replacement in emoji_replacements.items():
            safe_message = safe_message.replace(emoji, replacement)
        
        return safe_message
    
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
            'session_id': self.session_id,
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
        total_logs = sum(self.log_counts.values())
        
        return {
            'session_id': self.session_id,
            'runtime_seconds': runtime.total_seconds(),
            'log_counts': dict(self.log_counts),
            'total_logs': total_logs,
            'error_rate': (self.log_counts['error'] / max(total_logs, 1)) * 100,
            'warning_rate': (self.log_counts['warning'] / max(total_logs, 1)) * 100
        }
    
    def display_performance_summary(self):
        """Display performance summary"""
        summary = self.get_performance_summary()
        print("\n" + "="*60)
        print("📊 LOGGING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session ID: {summary['session_id']}")
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
    
    def save_session_report(self):
        """Save complete session report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"logs/performance/session_report_{timestamp}.json"
        
        report = {
            'session_summary': self.get_performance_summary(),
            'processes': dict(self.progress_tracker.processes),
            'error_counts': dict(self.progress_tracker.error_count),
            'warning_counts': dict(self.progress_tracker.warning_count)
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Failed to save session report: {e}")


# Global logger instance
_global_advanced_logger = None


def get_advanced_logger(name: str = "NICEGOLD_ADVANCED") -> EnterpriseAdvancedLogger:
    """Get global advanced logger instance"""
    global _global_advanced_logger
    if _global_advanced_logger is None:
        _global_advanced_logger = EnterpriseAdvancedLogger(name)
    return _global_advanced_logger


# Export main classes and functions
__all__ = [
    'EnterpriseAdvancedLogger',
    'AdvancedProgressTracker',
    'ProcessStatus',
    'LogLevel',
    'get_advanced_logger'
]
