#!/usr/bin/env python3
"""
🎨 ENHANCED BEAUTIFUL PROGRESS BAR AND LOGGING SYSTEM
ระบบ Progress Bar และ Logging ที่สวยงามพร้อมการแสดงผลแบบ Real-time

Enhanced Features:
- Multi-style progress bars (Classic, Modern, Neon, Enterprise, Rainbow)
- Real-time animated progress with ETA calculation
- Beautiful colored logging with icons and formatting
- Step-by-step process tracking
- Error handling and detailed reporting
- Enterprise-grade visual feedback
- Rich console integration with fallback support
"""

import time
import sys
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import threading
import logging
import queue
from datetime import datetime
import traceback

# Enhanced Rich import with better error handling
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn, 
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
        TaskProgressColumn, SpeedColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.status import Status
    from rich.tree import Tree
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("💡 Installing rich for beautiful progress bars...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich>=12.0.0"])
        from rich.console import Console
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn, 
            TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
            TaskProgressColumn, SpeedColumn
        )
        from rich.panel import Panel
        from rich.table import Table
        from rich.live import Live
        from rich.text import Text
        from rich.align import Align
        from rich import box
        from rich.status import Status
        from rich.tree import Tree
        from rich.columns import Columns
        RICH_AVAILABLE = True
    except Exception:
        RICH_AVAILABLE = False
    from rich import box


class StepStatus(Enum):
    """สถานะของแต่ละขั้นตอน"""
    PENDING = "⏳"
    RUNNING = "🔄"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    SKIPPED = "⏭️"


@dataclass
class PipelineStep:
    """ข้อมูลแต่ละขั้นตอนใน Pipeline"""
    id: int
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    sub_steps: List[str] = None
    current_sub_step: str = ""


class ProgressStyle(Enum):
    """สไตล์ Progress Bar ต่างๆ"""
    CLASSIC = "classic"
    MODERN = "modern" 
    NEON = "neon"
    ENTERPRISE = "enterprise"
    RAINBOW = "rainbow"
    MINIMAL = "minimal"


class LogLevel(Enum):
    """ระดับ Log พร้อมสี"""
    DEBUG = ("DEBUG", "dim white", "🔍")
    INFO = ("INFO", "blue", "ℹ️")
    SUCCESS = ("SUCCESS", "green", "✅")
    WARNING = ("WARNING", "yellow", "⚠️")
    ERROR = ("ERROR", "red", "❌")
    CRITICAL = ("CRITICAL", "bold red on white", "💥")


class EnhancedBeautifulLogger:
    """ระบบ Logging ที่สวยงามแบบ Enhanced"""
    
    def __init__(self, name: str = "NICEGOLD", use_rich: bool = True):
        self.name = name
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        
        # Fallback colors for non-rich environments
        self.colors = {
            'reset': '\033[0m', 'bold': '\033[1m', 'dim': '\033[2m',
            'green': '\033[92m', 'blue': '\033[94m', 'yellow': '\033[93m',
            'red': '\033[91m', 'cyan': '\033[96m', 'magenta': '\033[95m',
            'white': '\033[97m', 'bg_green': '\033[42m', 'bg_red': '\033[41m'
        }
        
    def _log_rich(self, level: LogLevel, message: str, details: Optional[Dict] = None):
        """Log ด้วย Rich library"""
        if not self.use_rich:
            return self._log_fallback(level, message, details)
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_name, color, icon = level.value
        
        # สร้าง panel สำหรับ log message
        content = Text()
        content.append(f"{icon} {timestamp} ", style="dim")
        content.append(f"[{level_name}] ", style=f"bold {color}")
        content.append(f"{self.name} ", style="bold cyan")
        content.append("→ ", style="dim")
        content.append(message, style=color)
        
        if details:
            content.append("\n")
            for key, value in details.items():
                content.append(f"    • {key}: ", style="dim")
                content.append(str(value), style="white")
                content.append("\n")
        
        # แสดง panel ที่สวยงาม
        if level == LogLevel.CRITICAL:
            panel = Panel(content, border_style="red", box=box.DOUBLE)
        elif level == LogLevel.ERROR:
            panel = Panel(content, border_style="red")
        elif level == LogLevel.WARNING:
            panel = Panel(content, border_style="yellow")
        elif level == LogLevel.SUCCESS:
            panel = Panel(content, border_style="green")
        else:
            panel = Panel(content, border_style="blue", box=box.MINIMAL)
            
        self.console.print(panel)
        
    def _log_fallback(self, level: LogLevel, message: str, details: Optional[Dict] = None):
        """Log แบบ fallback สำหรับสภาพแวดล้อมที่ไม่มี Rich"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_name, _, icon = level.value
        
        # เลือกสี
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            color = self.colors['red']
        elif level == LogLevel.WARNING:
            color = self.colors['yellow']
        elif level == LogLevel.SUCCESS:
            color = self.colors['green']
        elif level == LogLevel.INFO:
            color = self.colors['blue']
        else:
            color = self.colors['dim']
            
        # สร้างข้อความ
        header = f"{color}{self.colors['bold']}{icon} {timestamp} [{level_name}] {self.name}{self.colors['reset']}"
        msg_line = f"    {color}{message}{self.colors['reset']}"
        
        print(header)
        print(msg_line)
        
        if details:
            for key, value in details.items():
                print(f"    {self.colors['dim']}• {key}: {value}{self.colors['reset']}")
        print()
        
    def debug(self, message: str, details: Optional[Dict] = None):
        """Log DEBUG level"""
        self._log_rich(LogLevel.DEBUG, message, details)
        
    def info(self, message: str, details: Optional[Dict] = None):
        """Log INFO level"""
        self._log_rich(LogLevel.INFO, message, details)
        
    def success(self, message: str, details: Optional[Dict] = None):
        """Log SUCCESS level"""
        self._log_rich(LogLevel.SUCCESS, message, details)
        
    def warning(self, message: str, details: Optional[Dict] = None):
        """Log WARNING level"""
        self._log_rich(LogLevel.WARNING, message, details)
        
    def error(self, message: str, details: Optional[Dict] = None):
        """Log ERROR level"""
        self._log_rich(LogLevel.ERROR, message, details)
        
    def critical(self, message: str, details: Optional[Dict] = None):
        """Log CRITICAL level"""
        self._log_rich(LogLevel.CRITICAL, message, details)
        
    def step_start(self, step_number: int, step_name: str, description: str = ""):
        """เริ่มต้นขั้นตอนใหม่"""
        if self.use_rich:
            content = Text()
            content.append("🚀 STEP ", style="bold blue")
            content.append(f"{step_number}", style="bold cyan")
            content.append(": ", style="bold blue")
            content.append(step_name, style="bold white")
            if description:
                content.append("\n")
                content.append(description, style="dim")
                
            panel = Panel(
                content,
                border_style="blue",
                box=box.DOUBLE,
                padding=(0, 1)
            )
            self.console.print(panel)
        else:
            print(f"\n{self.colors['blue']}{self.colors['bold']}🚀 STEP {step_number}: {step_name}{self.colors['reset']}")
            if description:
                print(f"    {self.colors['dim']}{description}{self.colors['reset']}")
        
    def step_complete(self, step_number: int, step_name: str, duration: float, details: Optional[Dict] = None):
        """เสร็จสิ้นขั้นตอน"""
        duration_str = f"{duration:.2f}s"
        
        if self.use_rich:
            content = Text()
            content.append("✅ COMPLETED STEP ", style="bold green")
            content.append(f"{step_number}", style="bold cyan")
            content.append(": ", style="bold green")
            content.append(step_name, style="bold white")
            content.append(f" ({duration_str})", style="dim")
            
            if details:
                content.append("\n")
                for key, value in details.items():
                    content.append(f"    ✓ {key}: ", style="green")
                    content.append(str(value), style="white")
                    content.append("\n")
                    
            panel = Panel(
                content,
                border_style="green",
                padding=(0, 1)
            )
            self.console.print(panel)
        else:
            print(f"{self.colors['green']}{self.colors['bold']}✅ COMPLETED STEP {step_number}: {step_name} ({duration_str}){self.colors['reset']}")
            if details:
                for key, value in details.items():
                    print(f"    {self.colors['green']}✓ {key}: {value}{self.colors['reset']}")
        
    def step_error(self, step_number: int, step_name: str, error: str, details: Optional[Dict] = None):
        """ขั้นตอนผิดพลาด"""
        if self.use_rich:
            content = Text()
            content.append("❌ FAILED STEP ", style="bold red")
            content.append(f"{step_number}", style="bold cyan")
            content.append(": ", style="bold red")
            content.append(step_name, style="bold white")
            content.append("\n")
            content.append(f"Error: {error}", style="red")
            
            if details:
                content.append("\n")
                for key, value in details.items():
                    content.append(f"    • {key}: ", style="dim")
                    content.append(str(value), style="white")
                    content.append("\n")
                    
            panel = Panel(
                content,
                border_style="red",
                box=box.DOUBLE
            )
            self.console.print(panel)
        else:
            print(f"{self.colors['red']}{self.colors['bold']}❌ FAILED STEP {step_number}: {step_name}{self.colors['reset']}")
            print(f"    {self.colors['red']}Error: {error}{self.colors['reset']}")
            if details:
                for key, value in details.items():
                    print(f"    {self.colors['dim']}• {key}: {value}{self.colors['reset']}")


class EnhancedProgressBar:
    """Progress Bar แบบ Enhanced พร้อม Animation"""
    
    def __init__(self, total: int, description: str = "", style: ProgressStyle = ProgressStyle.ENTERPRISE):
        self.total = total
        self.current = 0
        self.description = description
        self.style = style
        self.start_time = time.time()
        self.use_rich = RICH_AVAILABLE
        
        if self.use_rich:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                MofNCompleteColumn(),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                SpeedColumn(),
                console=Console()
            )
            self.task_id = self.progress.add_task(description, total=total)
            self.progress.start()
        else:
            # Fallback progress bar
            self.width = 50
            self.colors = {
                'green': '\033[92m', 'blue': '\033[94m', 'yellow': '\033[93m',
                'reset': '\033[0m', 'bold': '\033[1m'
            }
    
    def update(self, advance: int = 1, description: str = None):
        """อัพเดท progress"""
        self.current += advance
        if self.current > self.total:
            self.current = self.total
            
        if self.use_rich:
            if description:
                self.progress.update(self.task_id, advance=advance, description=description)
            else:
                self.progress.update(self.task_id, advance=advance)
        else:
            self._update_fallback(description)
    
    def _update_fallback(self, message: str = None):
        """Fallback progress bar"""
        percentage = (self.current / self.total) * 100
        filled_length = int(self.width * self.current / self.total)
        bar = ('█' * filled_length + '░' * (self.width - filled_length))
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "--s"
        
        line = (f"\r{self.colors['blue']}{self.colors['bold']}{self.description}{self.colors['reset']} "
                f"[{bar}] {percentage:6.2f}% ({self.current}/{self.total}) ETA: {eta_str}")
        
        if message:
            line += f" | {message}"
            
        sys.stdout.write('\033[K' + line)
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()
    
    def finish(self, final_message: str = "Completed!"):
        """จบ progress bar"""
        if self.use_rich:
            self.progress.update(self.task_id, description=final_message)
            self.progress.stop()
        else:
            self.current = self.total
            self._update_fallback(final_message)
            print()


class BeautifulProgressTracker:
    """ระบบติดตาม Progress แบบสวยงาม"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.console = Console()
        self.steps: Dict[int, PipelineStep] = {}
        self.current_step_id: Optional[int] = None
        self.overall_progress = 0.0
        self.start_time = time.time()
        self.is_active = False
        
        # Initialize pipeline steps
        self._initialize_steps()
    
    def _initialize_steps(self):
        """เริ่มต้นขั้นตอนทั้งหมด"""
        pipeline_steps = [
            (1, "Data Loading", "Loading real market data from datacsv/", [
                "Scanning CSV files",
                "Validating file format",
                "Loading data to memory",
                "Initial data validation"
            ]),
            (2, "Data Preprocessing", "Cleaning and preprocessing market data", [
                "Removing duplicates",
                "Handling missing values",
                "Standardizing columns",
                "Validating OHLC data"
            ]),
            (3, "Elliott Wave Detection", "Detecting Elliott Wave patterns", [
                "Calculating pivot points",
                "Identifying wave structures",
                "Pattern validation",
                "Wave labeling"
            ]),
            (4, "Feature Engineering", "Creating advanced technical features", [
                "Technical indicators",
                "Elliott Wave features",
                "Price action features",
                "Volatility features"
            ]),
            (5, "Feature Selection", "SHAP + Optuna feature selection", [
                "Initial feature analysis",
                "SHAP importance calculation",
                "Optuna optimization",
                "Final feature selection"
            ]),
            (6, "CNN-LSTM Training", "Training CNN-LSTM Elliott Wave model", [
                "Data preparation",
                "Model architecture setup",
                "Training process",
                "Model validation"
            ]),
            (7, "DQN Training", "Training DQN reinforcement learning agent", [
                "Environment setup",
                "Agent initialization",
                "Training episodes",
                "Reward optimization"
            ]),
            (8, "Pipeline Integration", "Integrating all components", [
                "Component integration",
                "Cross-validation",
                "Performance testing",
                "System optimization"
            ]),
            (9, "Performance Analysis", "Analyzing system performance", [
                "Metrics calculation",
                "Performance validation",
                "Report generation",
                "Quality assessment"
            ]),
            (10, "Enterprise Validation", "Final enterprise compliance check", [
                "AUC validation",
                "Overfitting check",
                "Data leakage check",
                "Compliance report"
            ])
        ]
        
        for step_id, name, desc, sub_steps in pipeline_steps:
            self.steps[step_id] = PipelineStep(
                id=step_id,
                name=name,
                description=desc,
                sub_steps=sub_steps or []
            )
    
    def start_pipeline(self):
        """เริ่มต้น Pipeline"""
        self.is_active = True
        self.start_time = time.time()
        self.console.clear()
        
        # Display beautiful header
        self._display_header()
    
    def _display_header(self):
        """แสดงหัวข้อสวยงาม"""
        header_text = Text()
        header_text.append("🌊 ELLIOTT WAVE AI TRADING SYSTEM 🌊\n", style="bold cyan")
        header_text.append("Enterprise-Grade CNN-LSTM + DQN Pipeline\n", style="bold white")
        header_text.append("Real-Time Progress Tracking System", style="italic green")
        
        header_panel = Panel(
            Align.center(header_text),
            title="🚀 NICEGOLD ProjectP",
            subtitle="⚡ Powered by Enterprise AI",
            border_style="bright_blue",
            box=box.DOUBLE
        )
        
        self.console.print(header_panel)
        self.console.print()
    
    def start_step(self, step_id: int, custom_message: str = None):
        """เริ่มขั้นตอน"""
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.status = StepStatus.RUNNING
        step.start_time = time.time()
        step.progress = 0.0
        self.current_step_id = step_id
        
        message = custom_message or f"Starting {step.name}..."
        self.logger.info(f"🔄 Step {step_id}: {message}")
        
        # Update display
        self._update_display()
    
    def update_step_progress(self, step_id: int, progress: float, 
                           sub_step: str = "", message: str = ""):
        """อัปเดต Progress ของขั้นตอน"""
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.progress = min(100.0, max(0.0, progress))
        if sub_step:
            step.current_sub_step = sub_step
        
        # Update overall progress
        self._update_overall_progress()
        
        if message:
            self.logger.info(f"📊 Step {step_id}: {message}")
        
        # Update display
        self._update_display()
    
    def complete_step(self, step_id: int, success: bool = True, 
                     error_message: str = ""):
        """เสร็จสิ้นขั้นตอน"""
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.end_time = time.time()
        step.progress = 100.0
        
        if success:
            step.status = StepStatus.SUCCESS
            self.logger.info(f"✅ Step {step_id}: {step.name} completed successfully!")
        else:
            step.status = StepStatus.ERROR
            step.error_message = error_message
            self.logger.error(f"❌ Step {step_id}: {step.name} failed - {error_message}")
        
        # Update display
        self._update_display()
    
    def _update_overall_progress(self):
        """อัปเดต Progress รวม"""
        total_progress = sum(step.progress for step in self.steps.values())
        self.overall_progress = total_progress / len(self.steps)
    
    def _update_display(self):
        """อัปเดตการแสดงผล"""
        if not self.is_active:
            return
        
        # Create progress table
        table = Table(
            title="🔄 Pipeline Progress Status",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Step", style="bold", width=6)
        table.add_column("Status", width=8)
        table.add_column("Name", style="bold white", width=25)
        table.add_column("Progress", width=20)
        table.add_column("Current Task", style="italic", width=25)
        table.add_column("Time", width=10)
        
        for step in self.steps.values():
            # Status with emoji and color
            status_text = Text(f"{step.status.value}")
            if step.status == StepStatus.SUCCESS:
                status_text.stylize("bold green")
            elif step.status == StepStatus.ERROR:
                status_text.stylize("bold red")
            elif step.status == StepStatus.RUNNING:
                status_text.stylize("bold yellow")
            else:
                status_text.stylize("dim white")
            
            # Progress bar
            if step.status == StepStatus.RUNNING:
                progress_bar = f"[{'█' * int(step.progress/5)}{'░' * (20-int(step.progress/5))}] {step.progress:.1f}%"
                progress_text = Text(progress_bar, style="bold yellow")
            elif step.status == StepStatus.SUCCESS:
                progress_text = Text("[████████████████████] 100%", style="bold green")
            elif step.status == StepStatus.ERROR:
                progress_text = Text("[████████████████████] ERROR", style="bold red")
            else:
                progress_text = Text("[░░░░░░░░░░░░░░░░░░░░] 0%", style="dim white")
            
            # Current task
            current_task = step.current_sub_step if step.status == StepStatus.RUNNING else ""
            if step.status == StepStatus.ERROR and step.error_message:
                current_task = f"Error: {step.error_message[:20]}..."
            
            # Time calculation
            time_text = ""
            if step.start_time:
                if step.end_time:
                    elapsed = step.end_time - step.start_time
                    time_text = f"{elapsed:.1f}s"
                elif step.status == StepStatus.RUNNING:
                    elapsed = time.time() - step.start_time
                    time_text = f"{elapsed:.1f}s"
            
            table.add_row(
                f"Step {step.id}",
                status_text,
                step.name,
                progress_text,
                current_task,
                time_text
            )
        
        # Overall progress
        overall_bar = f"[{'█' * int(self.overall_progress/5)}{'░' * (20-int(self.overall_progress/5))}] {self.overall_progress:.1f}%"
        overall_text = Text(f"\n🎯 Overall Progress: {overall_bar}", style="bold cyan")
        
        # Elapsed time
        elapsed_total = time.time() - self.start_time
        time_info = Text(f"⏱️  Total Elapsed: {elapsed_total:.1f}s", style="bold white")
        
        # Clear and redraw
        self.console.clear()
        self._display_header()
        self.console.print(table)
        self.console.print(overall_text)
        self.console.print(time_info)
        self.console.print()
    
    def add_warning(self, step_id: int, warning_message: str):
        """เพิ่มคำเตือน"""
        if step_id in self.steps:
            step = self.steps[step_id]
            if step.status != StepStatus.ERROR:
                step.status = StepStatus.WARNING
            self.logger.warning(f"⚠️ Step {step_id}: {warning_message}")
    
    def complete_pipeline(self, success: bool = True):
        """เสร็จสิ้น Pipeline"""
        self.is_active = False
        total_time = time.time() - self.start_time
        
        if success:
            success_text = Text("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉", style="bold green")
            panel = Panel(
                Align.center(success_text),
                title="✅ SUCCESS",
                border_style="bright_green",
                box=box.DOUBLE
            )
        else:
            error_text = Text("💥 PIPELINE FAILED! 💥", style="bold red")
            panel = Panel(
                Align.center(error_text),
                title="❌ FAILURE",
                border_style="bright_red",
                box=box.DOUBLE
            )
        
        self.console.print(panel)
        self.console.print(f"\n⏱️  Total Pipeline Time: {total_time:.1f} seconds")
        
        # Final status summary
        success_count = sum(1 for step in self.steps.values() if step.status == StepStatus.SUCCESS)
        error_count = sum(1 for step in self.steps.values() if step.status == StepStatus.ERROR)
        warning_count = sum(1 for step in self.steps.values() if step.status == StepStatus.WARNING)
        
        summary_text = f"📊 Final Summary: {success_count} ✅ | {error_count} ❌ | {warning_count} ⚠️"
        self.console.print(summary_text, style="bold white")
    
    def get_step_summary(self) -> Dict[str, Any]:
        """ได้สรุปขั้นตอนทั้งหมด"""
        return {
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for step in self.steps.values() 
                                 if step.status == StepStatus.SUCCESS),
            "failed_steps": sum(1 for step in self.steps.values() 
                              if step.status == StepStatus.ERROR),
            "overall_progress": self.overall_progress,
            "total_time": time.time() - self.start_time if self.start_time else 0,
            "steps_detail": {
                step_id: {
                    "name": step.name,
                    "status": step.status.name,
                    "progress": step.progress,
                    "error_message": step.error_message
                }
                for step_id, step in self.steps.items()
            }
        }


# Global progress tracker instance
_progress_tracker: Optional[BeautifulProgressTracker] = None


def get_progress_tracker() -> BeautifulProgressTracker:
    """ได้ Progress Tracker แบบ Singleton"""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = BeautifulProgressTracker()
    return _progress_tracker


def start_pipeline_progress():
    """เริ่มต้น Pipeline Progress"""
    tracker = get_progress_tracker()
    tracker.start_pipeline()
    return tracker


def update_progress(step_id: int, progress: float, sub_step: str = "", message: str = ""):
    """อัปเดต Progress (Function Interface)"""
    tracker = get_progress_tracker()
    tracker.update_step_progress(step_id, progress, sub_step, message)


def complete_step(step_id: int, success: bool = True, error_message: str = ""):
    """เสร็จสิ้นขั้นตอน (Function Interface)"""
    tracker = get_progress_tracker()
    tracker.complete_step(step_id, success, error_message)


def start_step(step_id: int, custom_message: str = None):
    """เริ่มขั้นตอน (Function Interface)"""
    tracker = get_progress_tracker()
    tracker.start_step(step_id, custom_message)


# Demo function
if __name__ == "__main__":
    import random
    
    # Demo the progress tracker
    tracker = start_pipeline_progress()
    
    for step_id in range(1, 11):
        start_step(step_id)
        
        # Simulate work with sub-steps
        step = tracker.steps[step_id]
        for i, sub_step in enumerate(step.sub_steps):
            progress = (i + 1) / len(step.sub_steps) * 100
            update_progress(step_id, progress, sub_step, f"Processing {sub_step}...")
            time.sleep(random.uniform(0.5, 2.0))
        
        # Random success/failure for demo
        success = random.random() > 0.1  # 90% success rate
        if success:
            complete_step(step_id, True)
        else:
            complete_step(step_id, False, "Simulated error for demo")
            break
    
    tracker.complete_pipeline(True)
