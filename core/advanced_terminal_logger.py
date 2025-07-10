#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE ADVANCED TERMINAL LOGGER
ระบบ Logging ขั้นสูงสำหรับ Terminal แบบสวยงามและทันสมัย

🎯 Features:
- ✨ Beautiful Real-time Progress Bars
- 🎨 Advanced Color-coded Messages  
- 📊 Real-time Performance Monitoring
- 🛡️ Comprehensive Error Handling
- 💫 Modern Terminal UI
- 🔄 Process Status Tracking
- 📈 Live Statistics Dashboard
- 🎭 Rich Text Formatting
- ⚡ High-performance Logging
"""

import os
import sys
import time
import threading
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import platform
import psutil

# Rich library for beautiful terminal output
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.align import Align
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Installing fallback terminal logger...")

# Colorama for cross-platform colors
try:
    import colorama
    from colorama import Fore, Style, Back
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class LogLevel(Enum):
    """🎯 Enhanced Log Levels"""
    DEBUG = ("DEBUG", "🔍")
    INFO = ("INFO", "ℹ️")
    WARNING = ("WARNING", "⚠️")
    ERROR = ("ERROR", "❌")
    CRITICAL = ("CRITICAL", "🚨")
    SUCCESS = ("SUCCESS", "✅")
    PROGRESS = ("PROGRESS", "📊")
    SYSTEM = ("SYSTEM", "⚙️")
    PERFORMANCE = ("PERFORMANCE", "📈")
    SECURITY = ("SECURITY", "🛡️")
    DATA = ("DATA", "📊")
    AI = ("AI", "🧠")
    TRADE = ("TRADE", "💹")
    
    def __init__(self, name: str, emoji: str):
        self.level_name = name
        self.emoji = emoji


class ProcessStatus(Enum):
    """🔄 Process Status Types"""
    INITIALIZED = ("INITIALIZED", "🔄", "blue")
    STARTING = ("STARTING", "🚀", "yellow")
    RUNNING = ("RUNNING", "⚡", "green")
    PROCESSING = ("PROCESSING", "🔧", "cyan")
    COMPLETING = ("COMPLETING", "🎯", "magenta")
    SUCCESS = ("SUCCESS", "✅", "green")
    WARNING = ("WARNING", "⚠️", "yellow")
    ERROR = ("ERROR", "❌", "red")
    CRITICAL = ("CRITICAL", "🚨", "red")
    CANCELLED = ("CANCELLED", "🛑", "red")
    
    def __init__(self, name: str, emoji: str, color: str):
        self.status_name = name
        self.emoji = emoji
        self.color = color


class TerminalColors:
    """🎨 Enhanced Terminal Color Management"""
    
    # ANSI Color Codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Foreground Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright Colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background Colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    @classmethod
    def get_level_color(cls, level: LogLevel) -> str:
        """Get color for log level"""
        color_map = {
            LogLevel.DEBUG: cls.BRIGHT_BLACK,
            LogLevel.INFO: cls.BRIGHT_BLUE,
            LogLevel.WARNING: cls.BRIGHT_YELLOW,
            LogLevel.ERROR: cls.BRIGHT_RED,
            LogLevel.CRITICAL: cls.RED + cls.BG_WHITE + cls.BOLD,
            LogLevel.SUCCESS: cls.BRIGHT_GREEN,
            LogLevel.PROGRESS: cls.BRIGHT_CYAN,
            LogLevel.SYSTEM: cls.BRIGHT_MAGENTA,
            LogLevel.PERFORMANCE: cls.CYAN,
            LogLevel.SECURITY: cls.YELLOW,
            LogLevel.DATA: cls.BLUE,
            LogLevel.AI: cls.MAGENTA,
            LogLevel.TRADE: cls.GREEN
        }
        return color_map.get(level, cls.WHITE)


class RealTimeProgressBar:
    """📊 Real-time Progress Bar with Advanced Features"""
    
    def __init__(self, name: str, total: int = 100, 
                 width: int = 50, show_percentage: bool = True,
                 show_eta: bool = True, show_rate: bool = True):
        self.name = name
        self.total = total
        self.current = 0
        self.width = width
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.start_time = time.time()
        self.last_update = self.start_time
        self.rates = deque(maxlen=10)  # Store last 10 rates
        self.status = "⚡ Running"
        self.color = TerminalColors.BRIGHT_CYAN
        
    def update(self, increment: int = 1, status: str = None):
        """Update progress bar"""
        self.current = min(self.current + increment, self.total)
        current_time = time.time()
        
        # Calculate rate
        if current_time > self.last_update:
            rate = increment / (current_time - self.last_update)
            self.rates.append(rate)
        
        self.last_update = current_time
        if status:
            self.status = status
    
    def set_progress(self, value: int, status: str = None):
        """Set absolute progress value"""
        self.current = min(max(value, 0), self.total)
        if status:
            self.status = status
    
    def get_display(self) -> str:
        """Get formatted progress bar display"""
        if self.total == 0:
            return f"{self.name}: {self.status}"
        
        percentage = (self.current / self.total) * 100
        filled_width = int((self.current / self.total) * self.width)
        
        # Create progress bar
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        # Build display components
        components = [f"{self.name}:"]
        components.append(f"[{bar}]")
        
        if self.show_percentage:
            components.append(f"{percentage:.1f}%")
        
        components.append(f"({self.current}/{self.total})")
        
        if self.show_eta and self.current > 0:
            elapsed = time.time() - self.start_time
            if self.current < self.total:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = str(timedelta(seconds=int(eta)))
                components.append(f"ETA: {eta_str}")
        
        if self.show_rate and self.rates:
            avg_rate = sum(self.rates) / len(self.rates)
            components.append(f"Rate: {avg_rate:.1f}/s")
        
        components.append(f"- {self.status}")
        
        # Apply color
        display = " ".join(components)
        return f"{self.color}{display}{TerminalColors.RESET}"
    
    def complete(self, status: str = "✅ Completed"):
        """Mark progress as completed"""
        self.current = self.total
        self.status = status
        self.color = TerminalColors.BRIGHT_GREEN


class SystemMonitor:
    """📈 Real-time System Performance Monitor"""
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.log_count = defaultdict(int)
        self.error_count = 0
        self.warning_count = 0
        
    def update_stats(self, level: LogLevel):
        """Update monitoring statistics"""
        self.log_count[level] += 1
        
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self.error_count += 1
        elif level == LogLevel.WARNING:
            self.warning_count += 1
        
        # Update peak memory
        current_memory = self.process.memory_info().rss
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            # Memory in MB
            current_memory_mb = memory_info.rss / 1024 / 1024
            peak_memory_mb = self.peak_memory / 1024 / 1024
            initial_memory_mb = self.initial_memory / 1024 / 1024
            
            return {
                'uptime': uptime,
                'uptime_str': str(timedelta(seconds=int(uptime))),
                'cpu_percent': cpu_percent,
                'memory_current_mb': current_memory_mb,
                'memory_peak_mb': peak_memory_mb,
                'memory_initial_mb': initial_memory_mb,
                'memory_growth_mb': current_memory_mb - initial_memory_mb,
                'total_logs': sum(self.log_count.values()),
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'log_rate': sum(self.log_count.values()) / uptime if uptime > 0 else 0
            }
        except Exception as e:
            return {
                'error': f"Failed to get stats: {str(e)}",
                'uptime_str': 'Unknown'
            }


class ProcessTracker:
    """🔄 Advanced Process Tracking System"""
    
    def __init__(self):
        self.processes = {}
        self.lock = threading.Lock()
        self.next_process_id = 1
        
    def start_process(self, name: str, description: str = "", 
                     total_steps: int = 0) -> str:
        """Start tracking a new process"""
        with self.lock:
            process_id = f"proc_{self.next_process_id:04d}"
            self.next_process_id += 1
            
            self.processes[process_id] = {
                'id': process_id,
                'name': name,
                'description': description,
                'status': ProcessStatus.INITIALIZED,
                'start_time': datetime.now(),
                'end_time': None,
                'total_steps': total_steps,
                'current_step': 0,
                'progress_bar': RealTimeProgressBar(name, total_steps) if total_steps > 0 else None,
                'messages': [],
                'errors': [],
                'warnings': [],
                'duration': 0
            }
            
            return process_id
    
    def update_process(self, process_id: str, status: ProcessStatus = None,
                      step: int = None, message: str = None,
                      error: str = None, warning: str = None):
        """Update process information"""
        with self.lock:
            if process_id not in self.processes:
                return False
            
            process = self.processes[process_id]
            
            if status:
                process['status'] = status
            
            if step is not None:
                process['current_step'] = step
                if process['progress_bar']:
                    process['progress_bar'].set_progress(step, status.emoji if status else None)
            
            if message:
                process['messages'].append({
                    'timestamp': datetime.now(),
                    'message': message
                })
            
            if error:
                process['errors'].append({
                    'timestamp': datetime.now(),
                    'error': error
                })
            
            if warning:
                process['warnings'].append({
                    'timestamp': datetime.now(),
                    'warning': warning
                })
            
            return True
    
    def complete_process(self, process_id: str, success: bool = True,
                        final_message: str = None):
        """Complete a process"""
        with self.lock:
            if process_id not in self.processes:
                return False
            
            process = self.processes[process_id]
            process['end_time'] = datetime.now()
            process['duration'] = (process['end_time'] - process['start_time']).total_seconds()
            
            if success:
                process['status'] = ProcessStatus.SUCCESS
                if process['progress_bar']:
                    process['progress_bar'].complete()
            else:
                process['status'] = ProcessStatus.ERROR
            
            if final_message:
                process['messages'].append({
                    'timestamp': datetime.now(),
                    'message': final_message
                })
            
            return True
    
    def get_process(self, process_id: str) -> Optional[Dict]:
        """Get process information"""
        with self.lock:
            return self.processes.get(process_id)
    
    def get_active_processes(self) -> List[Dict]:
        """Get all active processes"""
        with self.lock:
            return [p for p in self.processes.values() 
                   if p['status'] not in [ProcessStatus.SUCCESS, ProcessStatus.ERROR, ProcessStatus.CANCELLED]]
    
    def get_progress_display(self, process_id: str) -> str:
        """Get formatted progress display for process"""
        process = self.get_process(process_id)
        if not process:
            return f"❌ Process {process_id} not found"
        
        if process['progress_bar']:
            return process['progress_bar'].get_display()
        else:
            status = process['status']
            return f"{status.emoji} {process['name']}: {status.status_name}"


class AdvancedTerminalLogger:
    """🚀 Advanced Terminal Logger with Rich Features"""
    
    def __init__(self, name: str = "NICEGOLD", 
                 enable_rich: bool = True,
                 enable_file_logging: bool = True,
                 log_dir: str = "logs",
                 max_console_lines: int = 1000):
        
        self.name = name
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.enable_file_logging = enable_file_logging
        self.max_console_lines = max_console_lines
        
        # Initialize components
        self.monitor = SystemMonitor()
        self.process_tracker = ProcessTracker()
        self.console_buffer = deque(maxlen=max_console_lines)
        self.lock = threading.Lock()
        
        # Setup Rich console if available
        if self.enable_rich:
            self.console = Console(
                width=120,
                color_system="auto",
                force_terminal=True,
                legacy_windows=False
            )
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
                transient=True
            )
        else:
            self.console = None
            self.progress = None
        
        # Setup file logging
        if enable_file_logging:
            self.setup_file_logging(log_dir)
        
        # Initialize session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = Path(log_dir) / f"terminal_session_{self.session_id}.log" if enable_file_logging else None
        
        # Print startup banner
        self._print_startup_banner()
    
    def setup_file_logging(self, log_dir: str):
        """Setup file logging directory structure"""
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (log_path / "errors").mkdir(exist_ok=True)
        (log_path / "warnings").mkdir(exist_ok=True)
        (log_path / "performance").mkdir(exist_ok=True)
        (log_path / "processes").mkdir(exist_ok=True)
    
    def _print_startup_banner(self):
        """Print beautiful startup banner"""
        if self.enable_rich and self.console:
            self._print_rich_banner()
        else:
            self._print_simple_banner()
    
    def _print_rich_banner(self):
        """Print Rich-formatted startup banner"""
        banner_text = f"""
🚀 NICEGOLD ENTERPRISE ADVANCED TERMINAL LOGGER
Session: {self.session_id}
Rich UI: ✅ Enabled | File Logging: {'✅' if self.enable_file_logging else '❌'}
System: {platform.system()} {platform.release()}
Python: {platform.python_version()}
        """.strip()
        
        panel = Panel(
            banner_text,
            title="🏢 Enterprise Terminal Logger",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()
    
    def _print_simple_banner(self):
        """Print simple text banner"""
        banner = f"""
{'='*80}
🚀 NICEGOLD ENTERPRISE ADVANCED TERMINAL LOGGER
{'='*80}
Session: {self.session_id}
Rich UI: ❌ Not Available | File Logging: {'✅' if self.enable_file_logging else '❌'}
System: {platform.system()} {platform.release()}
Python: {platform.python_version()}
{'='*80}
        """.strip()
        
        print(f"{TerminalColors.BRIGHT_CYAN}{banner}{TerminalColors.RESET}")
        print()
    
    def log(self, level: LogLevel, message: str, 
            category: str = "General", process_id: str = None,
            data: Dict = None, exception: Exception = None):
        """🎯 Main logging method with enhanced features"""
        
        timestamp = datetime.now()
        
        # Update monitoring stats
        self.monitor.update_stats(level)
        
        # Format log entry
        log_entry = self._format_log_entry(
            timestamp, level, message, category, 
            process_id, data, exception
        )
        
        # Add to console buffer
        with self.lock:
            self.console_buffer.append(log_entry)
        
        # Print to console
        self._print_to_console(log_entry, level)
        
        # Write to file
        if self.enable_file_logging:
            self._write_to_file(log_entry, level)
        
        # Update process if provided
        if process_id:
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.process_tracker.update_process(process_id, error=message)
            elif level == LogLevel.WARNING:
                self.process_tracker.update_process(process_id, warning=message)
            else:
                self.process_tracker.update_process(process_id, message=message)
    
    def _format_log_entry(self, timestamp: datetime, level: LogLevel,
                         message: str, category: str, process_id: str,
                         data: Dict, exception: Exception) -> Dict:
        """Format log entry"""
        entry = {
            'timestamp': timestamp.isoformat(),
            'level': level.level_name,
            'emoji': level.emoji,
            'message': message,
            'category': category,
            'process_id': process_id,
            'session_id': self.session_id
        }
        
        if data:
            entry['data'] = data
        
        if exception:
            entry['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        return entry
    
    def _print_to_console(self, log_entry: Dict, level: LogLevel):
        """Print log entry to console"""
        if self.enable_rich and self.console:
            self._print_rich_log(log_entry, level)
        else:
            self._print_simple_log(log_entry, level)
    
    def _print_rich_log(self, log_entry: Dict, level: LogLevel):
        """Print Rich-formatted log entry"""
        timestamp = datetime.fromisoformat(log_entry['timestamp'])
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        # Color mapping
        color_map = {
            LogLevel.DEBUG: "dim white",
            LogLevel.INFO: "bright_blue",
            LogLevel.WARNING: "bright_yellow", 
            LogLevel.ERROR: "bright_red",
            LogLevel.CRITICAL: "red on white",
            LogLevel.SUCCESS: "bright_green",
            LogLevel.PROGRESS: "bright_cyan",
            LogLevel.SYSTEM: "bright_magenta",
            LogLevel.PERFORMANCE: "cyan",
            LogLevel.SECURITY: "yellow",
            LogLevel.DATA: "blue",
            LogLevel.AI: "magenta",
            LogLevel.TRADE: "green"
        }
        
        color = color_map.get(level, "white")
        
        # Create styled text
        text = Text()
        text.append(f"{log_entry['emoji']} ", style="bold")
        text.append(f"{level.level_name} ", style=f"bold {color}")
        text.append(f"[{time_str}] ", style="dim")
        
        if log_entry.get('process_id'):
            text.append(f"[{log_entry['process_id']}] ", style="dim cyan")
        
        text.append(f"{log_entry['category']}: ", style="bold")
        text.append(log_entry['message'], style=color)
        
        # Add exception info if present
        if 'exception' in log_entry:
            text.append(f"\n💥 Exception: {log_entry['exception']['type']}: {log_entry['exception']['message']}", 
                       style="red")
        
        # Add data if present
        if 'data' in log_entry:
            text.append(f"\n📊 Data: {json.dumps(log_entry['data'], indent=2)}", 
                       style="dim")
        
        self.console.print(text)
    
    def _print_simple_log(self, log_entry: Dict, level: LogLevel):
        """Print simple text log entry"""
        timestamp = datetime.fromisoformat(log_entry['timestamp'])
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        color = TerminalColors.get_level_color(level)
        
        # Format message
        parts = [
            f"{log_entry['emoji']}",
            f"{level.level_name}",
            f"[{time_str}]"
        ]
        
        if log_entry.get('process_id'):
            parts.append(f"[{log_entry['process_id']}]")
        
        parts.extend([
            f"{log_entry['category']}:",
            log_entry['message']
        ])
        
        message = " ".join(parts)
        print(f"{color}{message}{TerminalColors.RESET}")
        
        # Add exception info if present
        if 'exception' in log_entry:
            print(f"{TerminalColors.BRIGHT_RED}💥 Exception: {log_entry['exception']['type']}: {log_entry['exception']['message']}{TerminalColors.RESET}")
        
        # Add data if present
        if 'data' in log_entry:
            print(f"{TerminalColors.DIM}📊 Data: {json.dumps(log_entry['data'], indent=2)}{TerminalColors.RESET}")
    
    def _write_to_file(self, log_entry: Dict, level: LogLevel):
        """Write log entry to file"""
        if not self.log_file_path:
            return
        
        try:
            # Write to main log file
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Write to specific log files for errors/warnings
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                error_file = self.log_file_path.parent / "errors" / f"errors_{self.session_id}.log"
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            elif level == LogLevel.WARNING:
                warning_file = self.log_file_path.parent / "warnings" / f"warnings_{self.session_id}.log"
                with open(warning_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
        
        except Exception as e:
            # Fallback: print to console if file writing fails
            print(f"❌ Failed to write to log file: {e}")
    
    # Convenience methods for different log levels
    def debug(self, message: str, category: str = "Debug", **kwargs):
        """Debug level logging"""
        self.log(LogLevel.DEBUG, message, category, **kwargs)
    
    def info(self, message: str, category: str = "Info", **kwargs):
        """Info level logging"""
        self.log(LogLevel.INFO, message, category, **kwargs)
    
    def warning(self, message: str, category: str = "Warning", **kwargs):
        """Warning level logging"""
        self.log(LogLevel.WARNING, message, category, **kwargs)
    
    def error(self, message: str, category: str = "Error", **kwargs):
        """Error level logging"""
        self.log(LogLevel.ERROR, message, category, **kwargs)
    
    def critical(self, message: str, category: str = "Critical", **kwargs):
        """Critical level logging"""
        self.log(LogLevel.CRITICAL, message, category, **kwargs)
    
    def success(self, message: str, category: str = "Success", **kwargs):
        """Success level logging"""
        self.log(LogLevel.SUCCESS, message, category, **kwargs)
    
    def progress(self, message: str, category: str = "Progress", **kwargs):
        """Progress level logging"""
        self.log(LogLevel.PROGRESS, message, category, **kwargs)
    
    def system(self, message: str, category: str = "System", **kwargs):
        """System level logging"""
        self.log(LogLevel.SYSTEM, message, category, **kwargs)
    
    def performance(self, message: str, category: str = "Performance", **kwargs):
        """Performance level logging"""
        self.log(LogLevel.PERFORMANCE, message, category, **kwargs)
    
    def security(self, message: str, category: str = "Security", **kwargs):
        """Security level logging"""
        self.log(LogLevel.SECURITY, message, category, **kwargs)
    
    def data_log(self, message: str, category: str = "Data", **kwargs):
        """Data level logging"""
        self.log(LogLevel.DATA, message, category, **kwargs)
    
    def ai_log(self, message: str, category: str = "AI", **kwargs):
        """AI level logging"""
        self.log(LogLevel.AI, message, category, **kwargs)
    
    def trade_log(self, message: str, category: str = "Trade", **kwargs):
        """Trade level logging"""
        self.log(LogLevel.TRADE, message, category, **kwargs)
    
    # Process management methods
    def start_process(self, name: str, description: str = "", 
                     total_steps: int = 0) -> str:
        """Start a new tracked process"""
        process_id = self.process_tracker.start_process(name, description, total_steps)
        self.info(f"Started process: {name}", "Process_Manager", process_id=process_id)
        return process_id
    
    def update_process(self, process_id: str, status: ProcessStatus = None,
                      step: int = None, message: str = None):
        """Update process status"""
        success = self.process_tracker.update_process(process_id, status, step, message)
        if not success:
            self.warning(f"Failed to update process: {process_id}", "Process_Manager")
    
    def complete_process(self, process_id: str, success: bool = True,
                        final_message: str = None):
        """Complete a process"""
        self.process_tracker.complete_process(process_id, success, final_message)
        status = "successfully" if success else "with errors"
        self.success(f"Process completed {status}", "Process_Manager", process_id=process_id)
    
    def get_progress_display(self, process_id: str) -> str:
        """Get progress display for a process"""
        return self.process_tracker.get_progress_display(process_id)
    
    def show_system_stats(self):
        """Display current system statistics"""
        stats = self.monitor.get_stats()
        
        if self.enable_rich and self.console:
            self._show_rich_stats(stats)
        else:
            self._show_simple_stats(stats)
    
    def _show_rich_stats(self, stats: Dict):
        """Show Rich-formatted system statistics"""
        table = Table(title="📈 System Performance Statistics", 
                     border_style="bright_blue")
        
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bright_white")
        table.add_column("Status", style="bold")
        
        # Uptime
        table.add_row("⏱️ Uptime", stats['uptime_str'], "✅ Running")
        
        # CPU
        cpu_status = "✅ Good" if stats.get('cpu_percent', 0) < 80 else "⚠️ High"
        table.add_row("🖥️ CPU Usage", f"{stats.get('cpu_percent', 0):.1f}%", cpu_status)
        
        # Memory
        memory_mb = stats.get('memory_current_mb', 0)
        memory_status = "✅ Good" if memory_mb < 500 else "⚠️ High"
        table.add_row("💾 Memory Usage", f"{memory_mb:.1f} MB", memory_status)
        table.add_row("📈 Peak Memory", f"{stats.get('memory_peak_mb', 0):.1f} MB", "")
        table.add_row("📊 Memory Growth", f"{stats.get('memory_growth_mb', 0):.1f} MB", "")
        
        # Logs
        table.add_row("📝 Total Logs", str(stats.get('total_logs', 0)), "")
        table.add_row("❌ Errors", str(stats.get('error_count', 0)), "")
        table.add_row("⚠️ Warnings", str(stats.get('warning_count', 0)), "")
        table.add_row("📊 Log Rate", f"{stats.get('log_rate', 0):.1f}/s", "")
        
        self.console.print(table)
    
    def _show_simple_stats(self, stats: Dict):
        """Show simple text system statistics"""
        print(f"\n{TerminalColors.BRIGHT_CYAN}📈 System Performance Statistics{TerminalColors.RESET}")
        print("=" * 50)
        print(f"⏱️ Uptime: {stats['uptime_str']}")
        print(f"🖥️ CPU Usage: {stats.get('cpu_percent', 0):.1f}%")
        print(f"💾 Memory Usage: {stats.get('memory_current_mb', 0):.1f} MB")
        print(f"📈 Peak Memory: {stats.get('memory_peak_mb', 0):.1f} MB")
        print(f"📊 Memory Growth: {stats.get('memory_growth_mb', 0):.1f} MB")
        print(f"📝 Total Logs: {stats.get('total_logs', 0)}")
        print(f"❌ Errors: {stats.get('error_count', 0)}")
        print(f"⚠️ Warnings: {stats.get('warning_count', 0)}")
        print(f"📊 Log Rate: {stats.get('log_rate', 0):.1f}/s")
        print("=" * 50)
    
    def export_session_log(self, export_path: str = None) -> str:
        """Export current session log to file"""
        if not export_path:
            export_path = f"terminal_session_export_{self.session_id}.json"
        
        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'system_stats': self.monitor.get_stats(),
            'console_buffer': list(self.console_buffer),
            'active_processes': self.process_tracker.get_active_processes()
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.success(f"Session log exported to: {export_path}", "Export")
            return export_path
        
        except Exception as e:
            self.error(f"Failed to export session log: {e}", "Export", exception=e)
            return None


# Global logger instance
terminal_logger = None

def get_terminal_logger() -> AdvancedTerminalLogger:
    """Get global terminal logger instance"""
    global terminal_logger
    if terminal_logger is None:
        terminal_logger = AdvancedTerminalLogger()
    return terminal_logger

def init_terminal_logger(name: str = "NICEGOLD", **kwargs) -> AdvancedTerminalLogger:
    """Initialize global terminal logger"""
    global terminal_logger
    terminal_logger = AdvancedTerminalLogger(name=name, **kwargs)
    return terminal_logger


# Example usage and testing
if __name__ == "__main__":
    # Initialize logger
    logger = AdvancedTerminalLogger("NICEGOLD_TEST")
    
    # Test different log levels
    logger.info("🚀 System starting up", "Startup")
    logger.success("✅ Configuration loaded", "Config")
    logger.warning("⚠️ Minor configuration issue detected", "Config")
    logger.error("❌ Database connection failed", "Database")
    logger.critical("🚨 Critical system error", "System")
    
    # Test process tracking
    process_id = logger.start_process("Data Processing", "Processing XAUUSD data", 100)
    
    for i in range(1, 101, 10):
        time.sleep(0.1)
        logger.update_process(process_id, step=i, message=f"Processing row {i}")
        logger.progress(f"Processed {i}/100 rows", "Data_Processing", process_id=process_id)
    
    logger.complete_process(process_id, True, "Data processing completed successfully")
    
    # Show system stats
    logger.show_system_stats()
    
    # Test AI and trading logs
    logger.ai_log("🧠 Neural network training started", "ML_Training")
    logger.trade_log("💹 Buy signal detected for XAUUSD", "Signal_Detection")
    
    # Export session
    logger.export_session_log()
    
    logger.success("🎉 Terminal logger test completed successfully!", "Test")
