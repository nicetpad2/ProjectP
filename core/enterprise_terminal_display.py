#!/usr/bin/env python3
"""
🎯 ENTERPRISE TERMINAL DISPLAY SYSTEM - NICEGOLD PRODUCTION
Advanced Terminal Interface with Beautiful UI, Process Tracking & Log Management

🏢 ENTERPRISE FEATURES:
- Modern Terminal UI with animations
- Real-time Process Bars with ETA calculations
- Intelligent Error Display with context
- Smart File Logging with rotation
- Production-grade status indicators
- Multi-level progress tracking
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

class ProcessStatus(Enum):
    """Process status enumeration"""
    INITIALIZING = "🔄"
    RUNNING = "⚡"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    COMPLETED = "🎉"
    PAUSED = "⏸️"
    CANCELLED = "🛑"

class LogLevel(Enum):
    """Enhanced logging levels"""
    DEBUG = "🐛"
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    ENTERPRISE = "🏢"
    PRODUCTION = "🚀"

@dataclass
class ProcessStep:
    """Individual process step tracking"""
    step_id: int
    name: str
    description: str
    status: ProcessStatus = ProcessStatus.INITIALIZING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None
    
    @property
    def eta(self) -> Optional[timedelta]:
        if self.progress > 0 and self.duration:
            total_time = self.duration / (self.progress / 100)
            return total_time - self.duration
        return None

class EnterpriseTerminalDisplay:
    """
    🏢 ENTERPRISE TERMINAL DISPLAY ENGINE
    Modern terminal interface with production-grade features
    """
    
    def __init__(self, project_name: str = "NICEGOLD ENTERPRISE", 
                 log_level: LogLevel = LogLevel.INFO):
        self.project_name = project_name
        self.log_level = log_level
        self.terminal_width = self._get_terminal_width()
        
        # Process tracking
        self.current_processes: Dict[str, ProcessStep] = {}
        self.completed_processes: List[ProcessStep] = []
        self.global_start_time = datetime.now()
        
        # Display state
        self.last_display_lines = 0
        self.animation_frame = 0
        self.update_lock = threading.Lock()
        
        # Colors and styles
        self.colors = {
            'header': '\033[1;96m',      # Bright cyan
            'success': '\033[1;92m',     # Bright green
            'warning': '\033[1;93m',     # Bright yellow
            'error': '\033[1;91m',       # Bright red
            'critical': '\033[1;95m',    # Bright magenta
            'info': '\033[1;94m',        # Bright blue
            'enterprise': '\033[1;97m',  # Bright white
            'production': '\033[1;96m',  # Bright cyan
            'reset': '\033[0m',          # Reset
            'bold': '\033[1m',           # Bold
            'dim': '\033[2m',            # Dim
        }
        
        # Progress bar styles
        self.progress_chars = {
            'filled': '█',
            'partial': ['▏', '▎', '▍', '▌', '▋', '▊', '▉'],
            'empty': '░'
        }
        
        # Animation frames
        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Initialize logging
        self.log_manager = EnterpriseLogManager()
        
        # Header display
        self._display_enterprise_header()
    
    def _get_terminal_width(self) -> int:
        """Get terminal width with fallback"""
        try:
            return os.get_terminal_size().columns
        except:
            return 120  # Enterprise default width
    
    def _display_enterprise_header(self):
        """Display beautiful enterprise header"""
        width = self.terminal_width
        
        # Clear screen
        print('\033[2J\033[H')
        
        # Header design
        header_lines = [
            "╭" + "─" * (width - 2) + "╮",
            f"│{' ' * ((width - len(self.project_name) - 4) // 2)}🏢 {self.project_name} 🏢{' ' * ((width - len(self.project_name) - 4) // 2)}│",
            f"│{' ' * ((width - 32) // 2)}🚀 ENTERPRISE PRODUCTION SYSTEM 🚀{' ' * ((width - 32) // 2)}│",
            f"│{' ' * ((width - 20) // 2)}⚡ REAL-TIME PROCESSING ⚡{' ' * ((width - 20) // 2)}│",
            "├" + "─" * (width - 2) + "┤",
            f"│ 📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' ' * (width - 35)}│",
            f"│ 🎯 Mode: Production Enterprise{' ' * (width - 30)}│",
            "╰" + "─" * (width - 2) + "╯"
        ]
        
        # Display with colors
        for line in header_lines:
            print(f"{self.colors['enterprise']}{line}{self.colors['reset']}")
        
        print()  # Space after header
    
    def create_process(self, process_id: str, name: str, description: str, 
                      total_items: int = 100) -> ProcessStep:
        """Create new process for tracking"""
        process = ProcessStep(
            step_id=len(self.current_processes) + 1,
            name=name,
            description=description,
            total_items=total_items,
            start_time=datetime.now()
        )
        
        with self.update_lock:
            self.current_processes[process_id] = process
        
        self.log_manager.log_event(LogLevel.PRODUCTION, f"Process Started: {name}", {
            'process_id': process_id,
            'description': description,
            'total_items': total_items
        })
        
        self._update_display()
        return process
    
    def update_process(self, process_id: str, progress: float = None, 
                      completed_items: int = None, status: ProcessStatus = None,
                      error_message: str = None, metadata: Dict = None):
        """Update process progress"""
        if process_id not in self.current_processes:
            return
        
        process = self.current_processes[process_id]
        
        if progress is not None:
            process.progress = min(100.0, max(0.0, progress))
        
        if completed_items is not None:
            process.completed_items = completed_items
            if process.total_items > 0:
                process.progress = (completed_items / process.total_items) * 100
        
        if status is not None:
            process.status = status
            
            if status in [ProcessStatus.SUCCESS, ProcessStatus.ERROR, 
                         ProcessStatus.COMPLETED, ProcessStatus.CANCELLED]:
                process.end_time = datetime.now()
                self.completed_processes.append(process)
                del self.current_processes[process_id]
        
        if error_message is not None:
            process.error_message = error_message
            process.status = ProcessStatus.ERROR
            
        if metadata is not None:
            process.metadata.update(metadata)
        
        self._update_display()
    
    def complete_process(self, process_id: str, success: bool = True, 
                        message: str = None):
        """Complete a process"""
        status = ProcessStatus.SUCCESS if success else ProcessStatus.ERROR
        self.update_process(process_id, progress=100.0, status=status, 
                          error_message=message if not success else None)
        
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        self.log_manager.log_event(level, f"Process Completed: {process_id}", {
            'success': success,
            'message': message
        })
    
    def display_error(self, title: str, error_message: str, 
                     context: Dict = None, suggestions: List[str] = None):
        """Display detailed error information"""
        width = self.terminal_width
        
        # Error box design
        error_lines = [
            "╭" + "─" * (width - 2) + "╮",
            f"│ 🚨 ENTERPRISE ERROR DETECTED 🚨{' ' * (width - 35)}│",
            "├" + "─" * (width - 2) + "┤",
            f"│ Title: {title}{' ' * (width - len(title) - 10)}│",
            f"│ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' ' * (width - 30)}│",
            "├" + "─" * (width - 2) + "┤"
        ]
        
        # Add error message (word wrapped)
        error_words = error_message.split()
        current_line = "│ Error: "
        for word in error_words:
            if len(current_line + word) > width - 3:
                error_lines.append(current_line + " " * (width - len(current_line) - 1) + "│")
                current_line = "│        " + word
            else:
                current_line += word + " "
        
        if current_line.strip() != "│":
            error_lines.append(current_line + " " * (width - len(current_line) - 1) + "│")
        
        # Add context if provided
        if context:
            error_lines.append("├" + "─" * (width - 2) + "┤")
            error_lines.append(f"│ Context:{' ' * (width - 11)}│")
            for key, value in context.items():
                line = f"│  • {key}: {value}"
                if len(line) > width - 3:
                    line = line[:width-6] + "..."
                error_lines.append(line + " " * (width - len(line) - 1) + "│")
        
        # Add suggestions if provided
        if suggestions:
            error_lines.append("├" + "─" * (width - 2) + "┤")
            error_lines.append(f"│ Suggestions:{' ' * (width - 15)}│")
            for suggestion in suggestions:
                line = f"│  💡 {suggestion}"
                if len(line) > width - 3:
                    line = line[:width-6] + "..."
                error_lines.append(line + " " * (width - len(line) - 1) + "│")
        
        error_lines.append("╰" + "─" * (width - 2) + "╯")
        
        # Display with error color
        print()
        for line in error_lines:
            print(f"{self.colors['error']}{line}{self.colors['reset']}")
        print()
        
        # Log the error
        self.log_manager.log_event(LogLevel.ERROR, title, {
            'error_message': error_message,
            'context': context,
            'suggestions': suggestions
        })
    
    def _update_display(self):
        """Update the terminal display"""
        with self.update_lock:
            # Clear previous lines
            if self.last_display_lines > 0:
                print(f'\033[{self.last_display_lines}A\033[J', end='')
            
            lines_printed = 0
            
            # Display current processes
            if self.current_processes:
                print(f"{self.colors['header']}📊 ACTIVE PROCESSES{self.colors['reset']}")
                lines_printed += 1
                
                for process_id, process in self.current_processes.items():
                    lines_printed += self._display_process(process)
                
                print()  # Space after processes
                lines_printed += 1
            
            # Display summary
            lines_printed += self._display_summary()
            
            self.last_display_lines = lines_printed
            self.animation_frame = (self.animation_frame + 1) % len(self.spinner_frames)
    
    def _display_process(self, process: ProcessStep) -> int:
        """Display individual process with progress bar"""
        lines = 0
        
        # Process header
        spinner = self.spinner_frames[self.animation_frame] if process.status == ProcessStatus.RUNNING else process.status.value
        duration_str = f"{process.duration.total_seconds():.1f}s" if process.duration else "0.0s"
        
        print(f"  {spinner} {self.colors['bold']}{process.name}{self.colors['reset']} ({duration_str})")
        lines += 1
        
        # Description
        print(f"     {self.colors['dim']}{process.description}{self.colors['reset']}")
        lines += 1
        
        # Progress bar
        if process.total_items > 0:
            progress_bar = self._create_progress_bar(process.progress, 50)
            eta_str = f"ETA: {process.eta.total_seconds():.0f}s" if process.eta else "ETA: --"
            
            print(f"     {progress_bar} {process.progress:5.1f}% ({process.completed_items}/{process.total_items}) {eta_str}")
            lines += 1
        
        # Error message if present
        if process.error_message:
            print(f"     {self.colors['error']}❌ {process.error_message}{self.colors['reset']}")
            lines += 1
        
        # Metadata if present
        if process.metadata:
            meta_items = []
            for key, value in list(process.metadata.items())[:3]:  # Show max 3 metadata items
                meta_items.append(f"{key}: {value}")
            if meta_items:
                print(f"     {self.colors['dim']}📊 {' | '.join(meta_items)}{self.colors['reset']}")
                lines += 1
        
        print()  # Space after each process
        lines += 1
        
        return lines
    
    def _create_progress_bar(self, progress: float, width: int = 50) -> str:
        """Create beautiful progress bar"""
        filled_width = int((progress / 100) * width)
        remainder = ((progress / 100) * width) - filled_width
        
        bar = self.colors['success'] + self.progress_chars['filled'] * filled_width
        
        if remainder > 0 and filled_width < width:
            partial_index = int(remainder * len(self.progress_chars['partial']))
            bar += self.progress_chars['partial'][min(partial_index, len(self.progress_chars['partial']) - 1)]
            filled_width += 1
        
        bar += self.colors['dim'] + self.progress_chars['empty'] * (width - filled_width)
        bar += self.colors['reset']
        
        return f"[{bar}]"
    
    def _display_summary(self) -> int:
        """Display session summary"""
        lines = 0
        
        total_duration = datetime.now() - self.global_start_time
        active_count = len(self.current_processes)
        completed_count = len(self.completed_processes)
        success_count = len([p for p in self.completed_processes if p.status == ProcessStatus.SUCCESS])
        error_count = len([p for p in self.completed_processes if p.status == ProcessStatus.ERROR])
        
        print(f"{self.colors['header']}📈 SESSION SUMMARY{self.colors['reset']}")
        lines += 1
        
        print(f"  ⏱️  Total Duration: {total_duration.total_seconds():.1f}s")
        print(f"  🔄 Active Processes: {active_count}")
        print(f"  ✅ Completed Successfully: {success_count}")
        print(f"  ❌ Failed: {error_count}")
        print(f"  📊 Total Processed: {completed_count}")
        lines += 5
        
        return lines
    
    def display_enterprise_summary(self, stage: str, summary: dict):
        """Display a beautiful, actionable enterprise summary box for a pipeline stage."""
        width = self.terminal_width
        title = f"🏢 ENTERPRISE SUMMARY: {stage.upper()}"
        box_lines = [
            "╭" + "─" * (width - 2) + "╮",
            f"│ {title}{' ' * (width - len(title) - 2)}│"
        ]
        # Add summary key metrics
        for key, value in summary.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
            else:
                value_str = str(value)
            line = f"│ {key}: {value_str}"
            if len(line) > width - 1:
                line = line[:width-4] + "..."
            box_lines.append(line + " " * (width - len(line) - 1) + "│")
        box_lines.append("╰" + "─" * (width - 2) + "╯")
        # Print with color
        print(f"{self.colors['success']}")
        for line in box_lines:
            print(line)
        print(f"{self.colors['reset']}")
        # Log as well
        self.log_manager.log_event(LogLevel.ENTERPRISE, f"Enterprise Summary: {stage}", summary)
        

class EnterpriseLogManager:
    """
    🏢 ENTERPRISE LOG MANAGEMENT SYSTEM
    Intelligent file logging with rotation and smart naming
    """
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory structure
        self.log_dirs = {
            'main': self.base_log_dir / "main",
            'errors': self.base_log_dir / "errors", 
            'processes': self.base_log_dir / "processes",
            'debug': self.base_log_dir / "debug",
            'enterprise': self.base_log_dir / "enterprise"
        }
        
        for log_dir in self.log_dirs.values():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.log_files = {
            'main': self._create_log_file('main', f"nicegold_main_{self.session_id}.log"),
            'errors': self._create_log_file('errors', f"errors_{self.session_id}.log"),
            'processes': self._create_log_file('processes', f"processes_{self.session_id}.log"),
            'debug': self._create_log_file('debug', f"debug_{self.session_id}.log"),
            'enterprise': self._create_log_file('enterprise', f"enterprise_{self.session_id}.log")
        }
        
        # Create symlinks to latest files
        self._create_latest_symlinks()
        
        # Initialize session
        self._initialize_session()
    
    def _create_log_file(self, category: str, filename: str) -> Path:
        """Create log file with proper headers"""
        file_path = self.log_dirs[category] / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# NICEGOLD ENTERPRISE LOG - {category.upper()}\n")
            f.write(f"# Session ID: {self.session_id}\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Category: {category}\n")
            f.write("# " + "="*80 + "\n\n")
        
        return file_path
    
    def _create_latest_symlinks(self):
        """Create symlinks to latest log files"""
        for category, file_path in self.log_files.items():
            latest_link = self.log_dirs[category] / f"latest_{category}.log"
            
            # Remove existing symlink
            if latest_link.is_symlink():
                latest_link.unlink()
            
            # Create new symlink
            try:
                latest_link.symlink_to(file_path.name)
            except:
                # Fallback for systems without symlink support
                pass
    
    def _initialize_session(self):
        """Initialize logging session"""
        self.log_event(LogLevel.ENTERPRISE, "Enterprise Logging System Initialized", {
            'session_id': self.session_id,
            'log_directories': {k: str(v) for k, v in self.log_dirs.items()},
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version.split()[0],
                'timestamp': datetime.now().isoformat()
            }
        })
    
    def log_event(self, level: LogLevel, message: str, data: Dict = None):
        """Log event to appropriate files"""
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'level': level.name,
            'emoji': level.value,
            'message': message,
            'data': data or {},
            'session_id': self.session_id
        }
        
        # Write to main log
        self._write_to_log('main', log_entry)
        
        # Write to specific category logs
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._write_to_log('errors', log_entry)
        
        if level == LogLevel.DEBUG:
            self._write_to_log('debug', log_entry)
        
        if level in [LogLevel.ENTERPRISE, LogLevel.PRODUCTION]:
            self._write_to_log('enterprise', log_entry)
        
        # Always write process events
        if 'process_id' in (data or {}):
            self._write_to_log('processes', log_entry)
    
    def _write_to_log(self, category: str, log_entry: Dict):
        """Write log entry to file"""
        try:
            with open(self.log_files[category], 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                f.flush()  # Ensure immediate write
        except Exception as e:
            # Fallback logging
            print(f"⚠️ Log write failed: {e}")
    
    def get_latest_logs(self, category: str = 'main', lines: int = 50) -> List[str]:
        """Get latest log entries"""
        try:
            with open(self.log_files[category], 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except:
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_dir in self.log_dirs.values():
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        log_file.unlink()
                        self.log_event(LogLevel.INFO, f"Cleaned up old log file: {log_file.name}")
                    except Exception as e:
                        self.log_event(LogLevel.WARNING, f"Failed to clean log file: {e}")

# Global instance for easy access
_terminal_display = None

def get_enterprise_terminal() -> EnterpriseTerminalDisplay:
    """Get or create enterprise terminal display instance"""
    global _terminal_display
    if _terminal_display is None:
        _terminal_display = EnterpriseTerminalDisplay()
    return _terminal_display

def log_enterprise_event(level: LogLevel, message: str, data: Dict = None):
    """Quick logging function"""
    terminal = get_enterprise_terminal()
    terminal.log_manager.log_event(level, message, data)

# Convenience functions
def create_process(process_id: str, name: str, description: str, total_items: int = 100):
    """Create new process tracking"""
    return get_enterprise_terminal().create_process(process_id, name, description, total_items)

def update_process(process_id: str, progress: float = None, completed_items: int = None, 
                  status: ProcessStatus = None, error_message: str = None, metadata: Dict = None):
    """Update process progress"""
    get_enterprise_terminal().update_process(process_id, progress, completed_items, 
                                           status, error_message, metadata)

def complete_process(process_id: str, success: bool = True, message: str = None):
    """Complete a process"""
    get_enterprise_terminal().complete_process(process_id, success, message)

def display_error(title: str, error_message: str, context: Dict = None, suggestions: List[str] = None):
    """Display detailed error"""
    get_enterprise_terminal().display_error(title, error_message, context, suggestions)

if __name__ == "__main__":
    # Demo of the enterprise terminal system
    terminal = EnterpriseTerminalDisplay("NICEGOLD ENTERPRISE DEMO")
    
    # Create demo processes
    terminal.create_process("data_load", "Loading Market Data", "Loading XAUUSD market data from CSV files", 1771970)
    time.sleep(1)
    
    terminal.create_process("feature_eng", "Feature Engineering", "Creating Elliott Wave technical indicators", 50)
    time.sleep(1)
    
    # Simulate progress
    for i in range(100):
        terminal.update_process("data_load", completed_items=i*17719)
        terminal.update_process("feature_eng", progress=i*2)
        time.sleep(0.1)
    
    # Complete processes
    terminal.complete_process("data_load", True, "All market data loaded successfully")
    terminal.complete_process("feature_eng", True, "All technical indicators created")
    
    # Show error demo
    terminal.display_error(
        "SHAP Analysis Failed",
        "Unable to complete SHAP feature importance analysis due to memory constraints",
        context={
            "dataset_size": "1,771,970 rows",
            "memory_usage": "89.5%",
            "available_memory": "2.1GB"
        },
        suggestions=[
            "Increase system memory allocation",
            "Use data sampling for SHAP analysis",
            "Enable memory optimization mode"
        ]
    )
    
    print(f"\n{terminal.colors['success']}✅ Enterprise Terminal Demo Completed!{terminal.colors['reset']}")
    print(f"{terminal.colors['info']}📄 Check logs in the 'logs/' directory{terminal.colors['reset']}")
