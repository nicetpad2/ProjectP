#!/usr/bin/env python3
"""
⚡ NICEGOLD ENTERPRISE REAL-TIME PROGRESS MANAGER
ระบบจัดการ Progress Bar แบบ Real-time สำหรับทั้งโปรเจค

🎯 Features:
- 🚀 Multi-threaded Progress Tracking
- 📊 Beautiful Real-time Progress Bars
- 🎨 Dynamic Color-coded Status
- 🔄 Nested Progress Support
- 📈 Performance Metrics Integration
- 🎭 Rich Terminal UI
- ⚡ High-performance Updates
- 🛡️ Error Recovery & Fallback
"""

import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from collections import defaultdict, deque
import math
import sys
import os

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.progress import (
        Progress, TaskID, BarColumn, TextColumn, 
        TimeRemainingColumn, SpinnerColumn, TimeElapsedColumn,
        MofNCompleteColumn, DownloadColumn, TransferSpeedColumn
    )
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressType(Enum):
    """📊 Progress Bar Types"""
    BASIC = ("Basic", "📊")
    DOWNLOAD = ("Download", "📥")
    UPLOAD = ("Upload", "📤")
    PROCESSING = ("Processing", "⚙️")
    TRAINING = ("Training", "🧠")
    ANALYSIS = ("Analysis", "🔍")
    OPTIMIZATION = ("Optimization", "🎯")
    VALIDATION = ("Validation", "✅")
    
    def __init__(self, name: str, emoji: str):
        self.type_name = name
        self.emoji = emoji


class ProgressStatus(Enum):
    """🔄 Progress Status"""
    INITIALIZING = ("INITIALIZING", "🔄", "blue")
    RUNNING = ("RUNNING", "⚡", "green")
    PAUSED = ("PAUSED", "⏸️", "yellow")
    COMPLETED = ("COMPLETED", "✅", "green")
    FAILED = ("FAILED", "❌", "red")
    CANCELLED = ("CANCELLED", "🛑", "red")
    
    def __init__(self, name: str, emoji: str, color: str):
        self.status_name = name
        self.emoji = emoji
        self.color = color


class ProgressMetrics:
    """📈 Progress Performance Metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.last_update = None
        self.update_count = 0
        self.speed_samples = deque(maxlen=10)
        self.eta_samples = deque(maxlen=5)
        
    def start(self):
        """Start tracking metrics"""
        self.start_time = time.time()
        self.last_update = self.start_time
        
    def update(self, current: int, total: int):
        """Update metrics with current progress"""
        current_time = time.time()
        self.update_count += 1
        
        if self.last_update and current > 0:
            time_diff = current_time - self.last_update
            if time_diff > 0:
                speed = 1 / time_diff  # Updates per second
                self.speed_samples.append(speed)
        
        # Calculate ETA
        if current > 0 and total > 0:
            elapsed = current_time - self.start_time
            progress_ratio = current / total
            if progress_ratio > 0:
                estimated_total_time = elapsed / progress_ratio
                eta = estimated_total_time - elapsed
                self.eta_samples.append(max(0, eta))
        
        self.last_update = current_time
    
    def get_speed(self) -> float:
        """Get average speed (updates/second)"""
        if not self.speed_samples:
            return 0.0
        return sum(self.speed_samples) / len(self.speed_samples)
    
    def get_eta(self) -> float:
        """Get estimated time to completion (seconds)"""
        if not self.eta_samples:
            return 0.0
        return sum(self.eta_samples) / len(self.eta_samples)
    
    def get_elapsed(self) -> float:
        """Get elapsed time (seconds)"""
        if not self.start_time:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def finish(self):
        """Mark metrics as finished"""
        self.end_time = time.time()


class RealTimeProgressBar:
    """🚀 Enhanced Real-time Progress Bar"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 total: int = 100,
                 progress_type: ProgressType = ProgressType.BASIC,
                 show_speed: bool = False,
                 show_eta: bool = True,
                 show_percentage: bool = True,
                 auto_refresh: bool = True):
        
        self.task_id = task_id
        self.name = name
        self.total = total
        self.current = 0
        self.progress_type = progress_type
        self.show_speed = show_speed
        self.show_eta = show_eta
        self.show_percentage = show_percentage
        self.auto_refresh = auto_refresh
        
        self.status = ProgressStatus.INITIALIZING
        self.metrics = ProgressMetrics()
        self.description = name
        self.lock = threading.Lock()
        
        # Rich progress task ID (if using Rich)
        self.rich_task_id = None
        
        # Error tracking
        self.errors = []
        self.warnings = []
        
    def start(self):
        """Start the progress bar"""
        with self.lock:
            self.status = ProgressStatus.RUNNING
            self.metrics.start()
    
    def update(self, increment: int = 1, description: str = None, force_refresh: bool = False):
        """Update progress"""
        with self.lock:
            old_current = self.current
            self.current = min(self.current + increment, self.total)
            
            if description:
                self.description = description
            
            # Update metrics
            self.metrics.update(self.current, self.total)
            
            # Auto-complete if reached total
            if self.current >= self.total and self.status == ProgressStatus.RUNNING:
                self.status = ProgressStatus.COMPLETED
                self.metrics.finish()
            
            return self.current != old_current or force_refresh
    
    def set_progress(self, value: int, description: str = None):
        """Set absolute progress value"""
        with self.lock:
            old_current = self.current
            self.current = min(max(value, 0), self.total)
            
            if description:
                self.description = description
            
            self.metrics.update(self.current, self.total)
            
            return self.current != old_current
    
    def pause(self):
        """Pause progress"""
        with self.lock:
            if self.status == ProgressStatus.RUNNING:
                self.status = ProgressStatus.PAUSED
    
    def resume(self):
        """Resume progress"""
        with self.lock:
            if self.status == ProgressStatus.PAUSED:
                self.status = ProgressStatus.RUNNING
    
    def complete(self, description: str = None):
        """Mark as completed"""
        with self.lock:
            self.current = self.total
            self.status = ProgressStatus.COMPLETED
            if description:
                self.description = description
            self.metrics.finish()
    
    def fail(self, error_message: str = None):
        """Mark as failed"""
        with self.lock:
            self.status = ProgressStatus.FAILED
            if error_message:
                self.errors.append({
                    'timestamp': datetime.now(),
                    'message': error_message
                })
            self.metrics.finish()
    
    def cancel(self):
        """Cancel progress"""
        with self.lock:
            self.status = ProgressStatus.CANCELLED
            self.metrics.finish()
    
    def add_warning(self, warning: str):
        """Add warning to progress"""
        with self.lock:
            self.warnings.append({
                'timestamp': datetime.now(),
                'message': warning
            })
    
    def get_percentage(self) -> float:
        """Get completion percentage"""
        if self.total == 0:
            return 100.0 if self.status == ProgressStatus.COMPLETED else 0.0
        return (self.current / self.total) * 100
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information"""
        with self.lock:
            return {
                'task_id': self.task_id,
                'name': self.name,
                'description': self.description,
                'current': self.current,
                'total': self.total,
                'percentage': self.get_percentage(),
                'status': self.status,
                'progress_type': self.progress_type,
                'elapsed': self.metrics.get_elapsed(),
                'eta': self.metrics.get_eta(),
                'speed': self.metrics.get_speed(),
                'errors': len(self.errors),
                'warnings': len(self.warnings)
            }
    
    def get_simple_display(self) -> str:
        """Get simple text display"""
        info = self.get_progress_info()
        percentage = info['percentage']
        
        # Create progress bar
        bar_width = 30
        filled = int((percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Status emoji
        status_emoji = info['status'].emoji
        
        # Build display
        parts = [
            f"{status_emoji} {info['name']}:",
            f"[{bar}]",
            f"{percentage:.1f}%",
            f"({info['current']}/{info['total']})"
        ]
        
        if info['eta'] > 0 and info['status'] == ProgressStatus.RUNNING:
            eta_str = str(timedelta(seconds=int(info['eta'])))
            parts.append(f"ETA: {eta_str}")
        
        if info['speed'] > 0:
            parts.append(f"Speed: {info['speed']:.1f}/s")
        
        if info['description'] != info['name']:
            parts.append(f"- {info['description']}")
        
        return " ".join(parts)


class NestedProgressGroup:
    """🔗 Nested Progress Group Manager"""
    
    def __init__(self, group_id: str, name: str):
        self.group_id = group_id
        self.name = name
        self.children = {}
        self.parent = None
        self.lock = threading.Lock()
        
    def add_child(self, progress_bar: RealTimeProgressBar):
        """Add child progress bar"""
        with self.lock:
            self.children[progress_bar.task_id] = progress_bar
    
    def remove_child(self, task_id: str):
        """Remove child progress bar"""
        with self.lock:
            if task_id in self.children:
                del self.children[task_id]
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress of all children"""
        with self.lock:
            if not self.children:
                return {
                    'percentage': 0.0,
                    'completed': 0,
                    'total': 0,
                    'active': 0
                }
            
            total_percentage = 0.0
            completed_count = 0
            active_count = 0
            
            for child in self.children.values():
                info = child.get_progress_info()
                total_percentage += info['percentage']
                
                if info['status'] == ProgressStatus.COMPLETED:
                    completed_count += 1
                elif info['status'] == ProgressStatus.RUNNING:
                    active_count += 1
            
            avg_percentage = total_percentage / len(self.children)
            
            return {
                'percentage': avg_percentage,
                'completed': completed_count,
                'total': len(self.children),
                'active': active_count
            }


class RealTimeProgressManager:
    """🎛️ Real-time Progress Manager"""
    
    def __init__(self, 
                 enable_rich: bool = True,
                 max_concurrent_bars: int = 10,
                 refresh_rate: float = 0.1,
                 auto_cleanup: bool = True):
        
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.max_concurrent_bars = max_concurrent_bars
        self.refresh_rate = refresh_rate
        self.auto_cleanup = auto_cleanup
        
        # Storage
        self.progress_bars = {}
        self.groups = {}
        self.active_bars = set()
        self.lock = threading.Lock()
        
        # Rich components
        if self.enable_rich:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TransferSpeedColumn(),
                console=self.console,
                auto_refresh=True,
                refresh_per_second=int(1/refresh_rate)
            )
            self.live = None
        else:
            self.console = None
            self.progress = None
            self.live = None
        
        # Update thread
        self.update_thread = None
        self.stop_updates = False
        
        # Statistics
        self.total_created = 0
        self.total_completed = 0
        self.total_failed = 0
        
    def start(self):
        """Start the progress manager"""
        if self.enable_rich:
            self.progress.start()
        
        # Start update thread for simple mode
        if not self.enable_rich and not self.update_thread:
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def stop(self):
        """Stop the progress manager"""
        self.stop_updates = True
        
        if self.enable_rich and self.progress:
            self.progress.stop()
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
    
    def create_progress(self, 
                       name: str,
                       total: int = 100,
                       progress_type: ProgressType = ProgressType.BASIC,
                       group_id: str = None,
                       **kwargs) -> str:
        """Create a new progress bar"""
        
        with self.lock:
            # Generate task ID
            task_id = f"task_{self.total_created:04d}_{int(time.time() * 1000) % 10000}"
            self.total_created += 1
            
            # Create progress bar
            progress_bar = RealTimeProgressBar(
                task_id=task_id,
                name=name,
                total=total,
                progress_type=progress_type,
                **kwargs
            )
            
            # Store progress bar
            self.progress_bars[task_id] = progress_bar
            self.active_bars.add(task_id)
            
            # Add to group if specified
            if group_id:
                if group_id not in self.groups:
                    self.groups[group_id] = NestedProgressGroup(group_id, f"Group {group_id}")
                self.groups[group_id].add_child(progress_bar)
            
            # Add to Rich progress if available
            if self.enable_rich and self.progress:
                rich_task_id = self.progress.add_task(
                    description=name,
                    total=total
                )
                progress_bar.rich_task_id = rich_task_id
            
            # Start the progress
            progress_bar.start()
            
            return task_id
    
    def update_progress(self, task_id: str, 
                       increment: int = 1, 
                       description: str = None,
                       total: int = None) -> bool:
        """Update progress bar"""
        
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            progress_bar = self.progress_bars[task_id]
            
            # Update total if provided
            if total is not None and total != progress_bar.total:
                progress_bar.total = total
                if self.enable_rich and progress_bar.rich_task_id is not None:
                    self.progress.update(progress_bar.rich_task_id, total=total)
            
            # Update progress
            changed = progress_bar.update(increment, description)
            
            # Update Rich progress
            if self.enable_rich and progress_bar.rich_task_id is not None and changed:
                update_kwargs = {
                    'advance': increment,
                }
                if description:
                    update_kwargs['description'] = description
                
                self.progress.update(progress_bar.rich_task_id, **update_kwargs)
            
            # Check if completed
            if progress_bar.status == ProgressStatus.COMPLETED:
                self._handle_completion(task_id, True)
            
            return True
    
    def set_progress(self, task_id: str, value: int, description: str = None) -> bool:
        """Set absolute progress value"""
        
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            progress_bar = self.progress_bars[task_id]
            old_current = progress_bar.current
            
            changed = progress_bar.set_progress(value, description)
            
            # Update Rich progress
            if self.enable_rich and progress_bar.rich_task_id is not None and changed:
                advance = progress_bar.current - old_current
                update_kwargs = {
                    'advance': advance,
                }
                if description:
                    update_kwargs['description'] = description
                
                self.progress.update(progress_bar.rich_task_id, **update_kwargs)
            
            # Check if completed
            if progress_bar.status == ProgressStatus.COMPLETED:
                self._handle_completion(task_id, True)
            
            return True
    
    def complete_progress(self, task_id: str, description: str = None) -> bool:
        """Mark progress as completed"""
        
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            progress_bar = self.progress_bars[task_id]
            progress_bar.complete(description)
            
            # Update Rich progress
            if self.enable_rich and progress_bar.rich_task_id is not None:
                remaining = progress_bar.total - progress_bar.current
                if remaining > 0:
                    self.progress.update(progress_bar.rich_task_id, advance=remaining)
                if description:
                    self.progress.update(progress_bar.rich_task_id, description=description)
            
            self._handle_completion(task_id, True)
            return True
    
    def fail_progress(self, task_id: str, error_message: str = None) -> bool:
        """Mark progress as failed"""
        
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            progress_bar = self.progress_bars[task_id]
            progress_bar.fail(error_message)
            
            # Update Rich progress
            if self.enable_rich and progress_bar.rich_task_id is not None:
                description = f"❌ {progress_bar.name}"
                if error_message:
                    description += f" - {error_message}"
                self.progress.update(progress_bar.rich_task_id, description=description)
            
            self._handle_completion(task_id, False)
            return True
    
    def pause_progress(self, task_id: str) -> bool:
        """Pause progress"""
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            self.progress_bars[task_id].pause()
            return True
    
    def resume_progress(self, task_id: str) -> bool:
        """Resume progress"""
        with self.lock:
            if task_id not in self.progress_bars:
                return False
            
            self.progress_bars[task_id].resume()
            return True
    
    def get_progress_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information"""
        with self.lock:
            if task_id not in self.progress_bars:
                return None
            return self.progress_bars[task_id].get_progress_info()
    
    def get_all_active_progress(self) -> List[Dict[str, Any]]:
        """Get all active progress information"""
        with self.lock:
            return [
                self.progress_bars[task_id].get_progress_info()
                for task_id in self.active_bars
                if task_id in self.progress_bars
            ]
    
    def _handle_completion(self, task_id: str, success: bool):
        """Handle progress completion"""
        if task_id in self.active_bars:
            self.active_bars.remove(task_id)
        
        if success:
            self.total_completed += 1
        else:
            self.total_failed += 1
        
        # Auto cleanup if enabled
        if self.auto_cleanup:
            # Remove from Rich progress after a delay
            if self.enable_rich and task_id in self.progress_bars:
                progress_bar = self.progress_bars[task_id]
                if progress_bar.rich_task_id is not None:
                    # Keep it visible for a moment
                    threading.Timer(2.0, lambda: self._cleanup_rich_task(progress_bar.rich_task_id)).start()
    
    def _cleanup_rich_task(self, rich_task_id):
        """Cleanup Rich task"""
        if self.enable_rich and self.progress:
            try:
                self.progress.remove_task(rich_task_id)
            except:
                pass  # Task might already be removed
    
    def _update_loop(self):
        """Update loop for simple mode"""
        last_display_time = 0
        
        while not self.stop_updates:
            try:
                current_time = time.time()
                
                # Update display every refresh_rate seconds
                if current_time - last_display_time >= self.refresh_rate:
                    self._update_simple_display()
                    last_display_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent high CPU
                
            except Exception as e:
                print(f"❌ Progress update error: {e}")
                time.sleep(1.0)
    
    def _update_simple_display(self):
        """Update simple text display"""
        active_progress = self.get_all_active_progress()
        
        if not active_progress:
            return
        
        # Clear previous lines (simple approach)
        # In a real implementation, you'd want to use proper terminal control
        for info in active_progress:
            task_id = info['task_id']
            if task_id in self.progress_bars:
                display = self.progress_bars[task_id].get_simple_display()
                print(f"\r{display}", end="", flush=True)
        
        if active_progress:
            print()  # New line after all progress bars
    
    def create_group(self, group_id: str, name: str) -> str:
        """Create a progress group"""
        with self.lock:
            if group_id not in self.groups:
                self.groups[group_id] = NestedProgressGroup(group_id, name)
            return group_id
    
    def get_group_progress(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group progress summary"""
        with self.lock:
            if group_id not in self.groups:
                return None
            return self.groups[group_id].get_overall_progress()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self.lock:
            return {
                'total_created': self.total_created,
                'total_completed': self.total_completed,
                'total_failed': self.total_failed,
                'active_count': len(self.active_bars),
                'total_bars': len(self.progress_bars),
                'groups_count': len(self.groups)
            }
    
    def cleanup_completed(self):
        """Clean up completed progress bars"""
        with self.lock:
            to_remove = []
            for task_id, progress_bar in self.progress_bars.items():
                if progress_bar.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                if task_id in self.progress_bars:
                    del self.progress_bars[task_id]
                if task_id in self.active_bars:
                    self.active_bars.remove(task_id)


# Context manager for easy progress tracking
class ProgressContext:
    """🎯 Context manager for progress tracking"""
    
    def __init__(self, manager: RealTimeProgressManager, 
                 name: str, total: int = 100, **kwargs):
        self.manager = manager
        self.name = name
        self.total = total
        self.kwargs = kwargs
        self.task_id = None
    
    def __enter__(self):
        self.task_id = self.manager.create_progress(self.name, self.total, **self.kwargs)
        return self.task_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task_id:
            if exc_type is None:
                self.manager.complete_progress(self.task_id, "✅ Completed successfully")
            else:
                error_msg = f"Exception: {exc_type.__name__}: {exc_val}"
                self.manager.fail_progress(self.task_id, error_msg)


# Global progress manager instance
progress_manager = None

def get_progress_manager() -> RealTimeProgressManager:
    """Get global progress manager instance"""
    global progress_manager
    if progress_manager is None:
        progress_manager = RealTimeProgressManager()
        progress_manager.start()
    return progress_manager

def init_progress_manager(**kwargs) -> RealTimeProgressManager:
    """Initialize global progress manager"""
    global progress_manager
    if progress_manager:
        progress_manager.stop()
    progress_manager = RealTimeProgressManager(**kwargs)
    progress_manager.start()
    return progress_manager


# Example usage and testing
if __name__ == "__main__":
    # Initialize progress manager
    manager = RealTimeProgressManager(enable_rich=RICH_AVAILABLE)
    manager.start()
    
    try:
        # Test basic progress
        task1 = manager.create_progress("Data Loading", 100, ProgressType.PROCESSING)
        
        for i in range(100):
            time.sleep(0.05)
            manager.update_progress(task1, 1, f"Loading row {i+1}")
        
        manager.complete_progress(task1, "✅ Data loaded successfully")
        
        # Test multiple concurrent progress bars
        tasks = []
        for i in range(3):
            task_id = manager.create_progress(f"Worker {i+1}", 50, ProgressType.TRAINING)
            tasks.append(task_id)
        
        # Update all tasks concurrently
        import random
        for step in range(50):
            for task_id in tasks:
                if random.random() > 0.1:  # 90% chance to update
                    manager.update_progress(task_id, 1, f"Processing step {step+1}")
            time.sleep(0.1)
        
        # Complete all tasks
        for i, task_id in enumerate(tasks):
            manager.complete_progress(task_id, f"Worker {i+1} finished")
        
        # Test context manager
        with ProgressContext(manager, "Context Test", 25) as task_id:
            for i in range(25):
                time.sleep(0.05)
                manager.update_progress(task_id, 1, f"Context step {i+1}")
        
        print("\n📊 Final Statistics:")
        stats = manager.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    finally:
        manager.stop()
        print("\n✅ Progress manager test completed!")
