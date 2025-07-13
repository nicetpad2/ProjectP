#!/usr/bin/env python3
"""
🏢 ENTERPRISE REAL-TIME DASHBOARD SYSTEM
Advanced real-time display, progress tracking, and error management

Features:
✅ Beautiful Real-time Progress Bars
✅ Advanced Error Management (Warning/Error/Critical)
✅ Smart File Management System
✅ Enterprise Session Tracking
✅ Live Performance Monitoring
✅ Production-grade Logging

Version: 1.0 Enterprise Edition
Date: 11 July 2025
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Rich library for beautiful terminal UI
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn, ProgressColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Utility imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class StatusLevel(Enum):
    """Enterprise status levels"""
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    DEBUG = "debug"

class ProgressType(Enum):
    """Types of progress tracking"""
    MAIN_PIPELINE = "main_pipeline"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    FEATURE_SELECTION = "feature_selection"
    VALIDATION = "validation"
    FILE_OPERATION = "file_operation"

@dataclass
class StatusMessage:
    """Enterprise status message structure"""
    level: StatusLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    component: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'details': self.details or {}
        }

@dataclass
class ProgressTask:
    """Enterprise progress task tracking"""
    task_id: str
    name: str
    total: int
    completed: int = 0
    task_type: ProgressType = ProgressType.MAIN_PIPELINE
    start_time: datetime = field(default_factory=datetime.now)
    eta: Optional[datetime] = None
    status: StatusLevel = StatusLevel.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnterpriseFileManager:
    """Smart file management with latest file tracking"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.base_path / "sessions" / self.session_id
        self.latest_files = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.session_path,
            self.session_path / "logs",
            self.session_path / "data",
            self.session_path / "models",
            self.session_path / "reports",
            self.session_path / "charts"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def register_file(self, file_type: str, file_path: Path, metadata: Dict[str, Any] = None) -> Path:
        """Register a new file with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate enterprise filename
        if file_type == "log":
            new_filename = f"enterprise_session_{timestamp}.log"
        elif file_type == "model":
            new_filename = f"model_{timestamp}.joblib"
        elif file_type == "report":
            new_filename = f"analysis_report_{timestamp}.json"
        elif file_type == "data":
            new_filename = f"processed_data_{timestamp}.csv"
        else:
            new_filename = f"{file_type}_{timestamp}{file_path.suffix}"
        
        target_path = self.session_path / file_type.lower() / new_filename
        
        # Register as latest file
        self.latest_files[file_type] = {
            'path': target_path,
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'session_id': self.session_id
        }
        
        return target_path
    
    def get_latest_file(self, file_type: str) -> Optional[Path]:
        """Get the latest file of specified type"""
        if file_type in self.latest_files:
            return self.latest_files[file_type]['path']
        return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        return {
            'session_id': self.session_id,
            'session_path': str(self.session_path),
            'latest_files': {k: str(v['path']) for k, v in self.latest_files.items()},
            'file_count': len(self.latest_files),
            'created': datetime.now().isoformat()
        }

class EnterpriseRealTimeDashboard:
    """
    🏢 Enterprise Real-Time Dashboard System
    
    Features:
    - Beautiful real-time progress bars
    - Advanced error management
    - Smart file tracking
    - Performance monitoring
    - Enterprise logging
    """
    
    def __init__(self, project_name: str = "NICEGOLD Enterprise ProjectP"):
        self.project_name = project_name
        self.console = Console() if RICH_AVAILABLE else None
        self.file_manager = EnterpriseFileManager(Path("outputs"))
        
        # Status tracking
        self.status_messages: List[StatusMessage] = []
        self.progress_tasks: Dict[str, ProgressTask] = {}
        self.is_running = False
        self.update_thread = None
        
        # Dashboard layout
        self.layout = Layout() if RICH_AVAILABLE else None
        self.live_display = None
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.performance_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'memory_total': 0.0,
            'tasks_completed': 0,
            'errors_count': 0,
            'warnings_count': 0
        }
        
        # Setup logging
        self._setup_enterprise_logging()
        
        # Initialize dashboard
        if RICH_AVAILABLE:
            self._setup_dashboard_layout()
    
    def _setup_enterprise_logging(self):
        """Setup enterprise-grade logging"""
        log_file = self.file_manager.register_file("log", Path("enterprise.log"))
        
        # Configure logger
        self.logger = logging.getLogger(f"enterprise_{self.file_manager.session_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_dashboard_layout(self):
        """Setup dashboard layout structure"""
        if not RICH_AVAILABLE:
            return
        
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        self.layout["left"].split_column(
            Layout(name="progress", ratio=1),
            Layout(name="status", ratio=1)
        )
        
        self.layout["footer"].split_row(
            Layout(name="performance", ratio=1),
            Layout(name="files", ratio=1)
        )
    
    def start_dashboard(self):
        """Start the real-time dashboard"""
        if not RICH_AVAILABLE:
            print("⚠️ Rich library not available. Using fallback display.")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_performance_metrics)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.live_display = Live(self.layout, refresh_per_second=4, screen=True)
        self.live_display.start()
        
        self.log_status(StatusLevel.SUCCESS, "Enterprise Dashboard started successfully")
    
    def stop_dashboard(self):
        """Stop the real-time dashboard"""
        self.is_running = False
        
        if self.live_display:
            self.live_display.stop()
        
        self.log_status(StatusLevel.INFO, "Enterprise Dashboard stopped")
    
    def _update_performance_metrics(self):
        """Update performance metrics in background"""
        while self.is_running:
            try:
                if PSUTIL_AVAILABLE:
                    self.performance_metrics['cpu_usage'] = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    self.performance_metrics['memory_usage'] = memory.used / 1024**3  # GB
                    self.performance_metrics['memory_total'] = memory.total / 1024**3  # GB
                
                # Update dashboard
                if RICH_AVAILABLE and self.live_display:
                    self._update_dashboard_content()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.log_status(StatusLevel.ERROR, f"Performance monitoring error: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _update_dashboard_content(self):
        """Update dashboard content"""
        if not RICH_AVAILABLE:
            return
        
        # Header
        self.layout["header"].update(self._create_header())
        
        # Progress section
        self.layout["progress"].update(self._create_progress_panel())
        
        # Status section
        self.layout["status"].update(self._create_status_panel())
        
        # Performance section
        self.layout["performance"].update(self._create_performance_panel())
        
        # Files section
        self.layout["files"].update(self._create_files_panel())
        
        # Right side - System info
        self.layout["right"].update(self._create_system_info_panel())
    
    def _create_header(self) -> Panel:
        """Create header panel"""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        header_text = Text()
        header_text.append("🏢 ", style="bold blue")
        header_text.append(self.project_name, style="bold white")
        header_text.append(f" | Session: {self.file_manager.session_id} | Uptime: {uptime_str}", style="dim")
        
        return Panel(
            Align.center(header_text),
            style="bold blue",
            box=box.ROUNDED
        )
    
    def _create_progress_panel(self) -> Panel:
        """Create progress tracking panel"""
        if not self.progress_tasks:
            content = Text("No active tasks", style="dim")
        else:
            content = Table(show_header=True, header_style="bold magenta")
            content.add_column("Task", style="cyan", no_wrap=True)
            content.add_column("Progress", justify="center")
            content.add_column("Status", justify="center")
            content.add_column("ETA", justify="center")
            
            for task in self.progress_tasks.values():
                progress_pct = (task.completed / task.total * 100) if task.total > 0 else 0
                
                # Progress bar
                bar_width = 20
                filled = int(bar_width * progress_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                # Status color
                status_color = {
                    StatusLevel.SUCCESS: "green",
                    StatusLevel.INFO: "blue",
                    StatusLevel.WARNING: "yellow",
                    StatusLevel.ERROR: "red",
                    StatusLevel.CRITICAL: "bold red"
                }.get(task.status, "white")
                
                # ETA calculation
                if task.completed > 0 and task.total > task.completed:
                    elapsed = datetime.now() - task.start_time
                    rate = task.completed / elapsed.total_seconds()
                    remaining = (task.total - task.completed) / rate
                    eta = "~" + str(timedelta(seconds=int(remaining))).split('.')[0]
                else:
                    eta = "N/A"
                
                content.add_row(
                    task.name[:20],
                    f"{bar} {progress_pct:.1f}%",
                    Text(task.status.value.upper(), style=status_color),
                    eta
                )
        
        return Panel(content, title="📊 Active Tasks", border_style="blue")
    
    def _create_status_panel(self) -> Panel:
        """Create status messages panel"""
        # Get recent messages (last 10)
        recent_messages = self.status_messages[-10:] if self.status_messages else []
        
        if not recent_messages:
            content = Text("No status messages", style="dim")
        else:
            content = Table(show_header=False, show_edge=False, pad_edge=False)
            content.add_column("Time", style="dim", width=8)
            content.add_column("Level", width=8)
            content.add_column("Message", ratio=1)
            
            for msg in recent_messages:
                time_str = msg.timestamp.strftime("%H:%M:%S")
                
                level_style = {
                    StatusLevel.SUCCESS: "bold green",
                    StatusLevel.INFO: "blue",
                    StatusLevel.WARNING: "yellow",
                    StatusLevel.ERROR: "red",
                    StatusLevel.CRITICAL: "bold red",
                    StatusLevel.DEBUG: "dim"
                }.get(msg.level, "white")
                
                level_icon = {
                    StatusLevel.SUCCESS: "✅",
                    StatusLevel.INFO: "ℹ️",
                    StatusLevel.WARNING: "⚠️",
                    StatusLevel.ERROR: "❌",
                    StatusLevel.CRITICAL: "🚨",
                    StatusLevel.DEBUG: "🔍"
                }.get(msg.level, "•")
                
                content.add_row(
                    time_str,
                    Text(f"{level_icon} {msg.level.value.upper()}", style=level_style),
                    msg.message[:60] + ("..." if len(msg.message) > 60 else "")
                )
        
        # Count messages by level
        level_counts = {}
        for msg in self.status_messages:
            level_counts[msg.level] = level_counts.get(msg.level, 0) + 1
        
        title = "📝 Status Messages"
        if level_counts:
            error_count = level_counts.get(StatusLevel.ERROR, 0) + level_counts.get(StatusLevel.CRITICAL, 0)
            warning_count = level_counts.get(StatusLevel.WARNING, 0)
            if error_count > 0:
                title += f" (🚨 {error_count} errors)"
            elif warning_count > 0:
                title += f" (⚠️ {warning_count} warnings)"
        
        return Panel(content, title=title, border_style="yellow")
    
    def _create_performance_panel(self) -> Panel:
        """Create performance monitoring panel"""
        content = Table(show_header=False, show_edge=False)
        content.add_column("Metric", style="cyan", width=15)
        content.add_column("Value", justify="right")
        content.add_column("Visual", width=20)
        
        # CPU Usage
        cpu = self.performance_metrics['cpu_usage']
        cpu_bar = self._create_mini_bar(cpu, 100)
        cpu_style = "green" if cpu < 70 else "yellow" if cpu < 90 else "red"
        content.add_row(
            "CPU Usage",
            Text(f"{cpu:.1f}%", style=cpu_style),
            cpu_bar
        )
        
        # Memory Usage
        mem_used = self.performance_metrics['memory_usage']
        mem_total = self.performance_metrics['memory_total']
        if mem_total > 0:
            mem_pct = (mem_used / mem_total) * 100
            mem_bar = self._create_mini_bar(mem_pct, 100)
            mem_style = "green" if mem_pct < 70 else "yellow" if mem_pct < 90 else "red"
            content.add_row(
                "Memory Usage",
                Text(f"{mem_used:.1f}/{mem_total:.1f}GB", style=mem_style),
                mem_bar
            )
        
        # Tasks completed
        tasks_completed = self.performance_metrics['tasks_completed']
        content.add_row(
            "Tasks Done",
            Text(str(tasks_completed), style="green"),
            "✅" * min(tasks_completed, 10)
        )
        
        # Error/Warning counts
        errors = self.performance_metrics['errors_count']
        warnings = self.performance_metrics['warnings_count']
        
        if errors > 0:
            content.add_row(
                "Errors",
                Text(str(errors), style="red"),
                "🚨" * min(errors, 10)
            )
        
        if warnings > 0:
            content.add_row(
                "Warnings",
                Text(str(warnings), style="yellow"),
                "⚠️" * min(warnings, 10)
            )
        
        return Panel(content, title="📈 Performance Monitor", border_style="green")
    
    def _create_files_panel(self) -> Panel:
        """Create file management panel"""
        content = Table(show_header=True, header_style="bold cyan")
        content.add_column("Type", style="cyan")
        content.add_column("Latest File", style="white")
        content.add_column("Time", style="dim")
        
        if not self.file_manager.latest_files:
            content.add_row("No files", "registered yet", "")
        else:
            for file_type, file_info in self.file_manager.latest_files.items():
                filename = file_info['path'].name
                time_str = file_info['timestamp'].strftime("%H:%M:%S")
                
                content.add_row(
                    file_type.upper(),
                    filename[:30] + ("..." if len(filename) > 30 else ""),
                    time_str
                )
        
        return Panel(content, title="📁 Latest Files", border_style="cyan")
    
    def _create_system_info_panel(self) -> Panel:
        """Create system information panel"""
        content = []
        
        # Session info
        content.append(Text("📋 Session Information", style="bold cyan"))
        content.append(Text(f"ID: {self.file_manager.session_id}", style="white"))
        content.append(Text(f"Started: {self.start_time.strftime('%H:%M:%S')}", style="dim"))
        content.append("")
        
        # System stats
        if PSUTIL_AVAILABLE:
            content.append(Text("💻 System Status", style="bold cyan"))
            content.append(Text("CPU Cores: " + str(psutil.cpu_count()), style="white"))
            
            # Memory info
            if self.performance_metrics['memory_total'] > 0:
                mem_pct = (self.performance_metrics['memory_usage'] / 
                          self.performance_metrics['memory_total']) * 100
                content.append(Text(f"RAM: {mem_pct:.1f}% used", style="white"))
            
            content.append("")
        
        # Active tasks summary
        content.append(Text("🎯 Tasks Summary", style="bold cyan"))
        content.append(Text(f"Active: {len(self.progress_tasks)}", style="white"))
        content.append(Text(f"Messages: {len(self.status_messages)}", style="dim"))
        
        return Panel(
            Align.left("\n".join(str(line) for line in content)),
            title="🔧 System Info",
            border_style="magenta"
        )
    
    def _create_mini_bar(self, value: float, max_value: float, width: int = 15) -> Text:
        """Create a mini progress bar"""
        if max_value == 0:
            return Text("░" * width, style="dim")
        
        filled = int(width * value / max_value)
        bar = "█" * filled + "░" * (width - filled)
        
        if value < 50:
            style = "green"
        elif value < 80:
            style = "yellow"
        else:
            style = "red"
        
        return Text(bar, style=style)
    
    # Public API Methods
    def log_status(self, level: StatusLevel, message: str, component: str = None, details: Dict[str, Any] = None):
        """Log a status message"""
        status_msg = StatusMessage(
            level=level,
            message=message,
            component=component,
            details=details
        )
        
        self.status_messages.append(status_msg)
        
        # Update metrics
        if level in [StatusLevel.ERROR, StatusLevel.CRITICAL]:
            self.performance_metrics['errors_count'] += 1
        elif level == StatusLevel.WARNING:
            self.performance_metrics['warnings_count'] += 1
        
        # Log to file
        log_level = {
            StatusLevel.DEBUG: logging.DEBUG,
            StatusLevel.INFO: logging.INFO,
            StatusLevel.WARNING: logging.WARNING,
            StatusLevel.ERROR: logging.ERROR,
            StatusLevel.CRITICAL: logging.CRITICAL,
            StatusLevel.SUCCESS: logging.INFO
        }.get(level, logging.INFO)
        
        log_message = message
        if component:
            log_message = f"[{component}] {log_message}"
        
        self.logger.log(log_level, log_message)
        
        # Keep only recent messages in memory
        if len(self.status_messages) > 1000:
            self.status_messages = self.status_messages[-500:]
    
    def create_progress_task(self, task_id: str, name: str, total: int, 
                           task_type: ProgressType = ProgressType.MAIN_PIPELINE) -> str:
        """Create a new progress task"""
        task = ProgressTask(
            task_id=task_id,
            name=name,
            total=total,
            task_type=task_type
        )
        
        self.progress_tasks[task_id] = task
        self.log_status(StatusLevel.INFO, f"Started task: {name}")
        
        return task_id
    
    def update_progress(self, task_id: str, completed: int, status: StatusLevel = None):
        """Update progress for a task"""
        if task_id not in self.progress_tasks:
            return
        
        task = self.progress_tasks[task_id]
        task.completed = completed
        
        if status:
            task.status = status
        
        # Calculate ETA
        if task.completed > 0 and task.total > task.completed:
            elapsed = datetime.now() - task.start_time
            rate = task.completed / elapsed.total_seconds()
            remaining_seconds = (task.total - task.completed) / rate
            task.eta = datetime.now() + timedelta(seconds=remaining_seconds)
    
    def complete_task(self, task_id: str, status: StatusLevel = StatusLevel.SUCCESS):
        """Complete a task"""
        if task_id not in self.progress_tasks:
            return
        
        task = self.progress_tasks[task_id]
        task.completed = task.total
        task.status = status
        
        self.performance_metrics['tasks_completed'] += 1
        self.log_status(status, f"Completed task: {task.name}")
        
        # Remove from active tasks after a delay
        def remove_task():
            time.sleep(2)
            if task_id in self.progress_tasks:
                del self.progress_tasks[task_id]
        
        threading.Thread(target=remove_task, daemon=True).start()
    
    def register_file(self, file_type: str, file_path: Path, metadata: Dict[str, Any] = None) -> Path:
        """Register a new file"""
        target_path = self.file_manager.register_file(file_type, file_path, metadata)
        self.log_status(StatusLevel.SUCCESS, f"Registered {file_type} file: {target_path.name}")
        return target_path
    
    def get_latest_file(self, file_type: str) -> Optional[Path]:
        """Get latest file of specified type"""
        return self.file_manager.get_latest_file(file_type)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        summary = self.file_manager.get_session_summary()
        summary.update({
            'performance_metrics': self.performance_metrics.copy(),
            'total_messages': len(self.status_messages),
            'active_tasks': len(self.progress_tasks),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        })
        return summary

# Factory function
def get_enterprise_dashboard(project_name: str = "NICEGOLD Enterprise ProjectP") -> EnterpriseRealTimeDashboard:
    """Get enterprise dashboard instance"""
    return EnterpriseRealTimeDashboard(project_name)

# Example usage and testing
if __name__ == "__main__":
    # Demo the dashboard system
    dashboard = get_enterprise_dashboard()
    
    try:
        dashboard.start_dashboard()
        
        # Simulate some work
        task_id = dashboard.create_progress_task("demo_task", "Demo Processing", 100)
        
        for i in range(101):
            dashboard.update_progress(task_id, i)
            
            if i == 25:
                dashboard.log_status(StatusLevel.WARNING, "This is a warning message")
            elif i == 50:
                dashboard.log_status(StatusLevel.ERROR, "This is an error message")
            elif i == 75:
                dashboard.log_status(StatusLevel.SUCCESS, "Checkpoint reached successfully")
            
            time.sleep(0.1)
        
        dashboard.complete_task(task_id)
        
        # Show final summary
        print(json.dumps(dashboard.get_session_summary(), indent=2))
        
        time.sleep(5)  # Display for 5 seconds
        
    finally:
        dashboard.stop_dashboard() 