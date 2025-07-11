1#!/usr/bin/env python3
"""
🏢 NICEGOLD Enterprise ProjectP - Enterprise Installation Manager
Advanced installation system with enterprise features and full dependency management
"""

import subprocess
import sys
import os
import platform
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def install_rich_if_missing():
    """Install rich library if missing"""
    try:
        import rich
        from rich.console import Console
        return Console()
    except ImportError:
        print("📦 Installing 'rich' library for enterprise UI...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rich'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from rich.console import Console
            return Console()
        except Exception as e:
            print(f"❌ Could not install 'rich': {e}")
            return None

console = install_rich_if_missing()

if console:
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.text import Text
    from rich.prompt import Confirm, Prompt

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements for NICEGOLD ProjectP"""
    requirements = {}
    
    # Check Python version
    python_version = sys.version_info
    requirements['python_version'] = python_version >= (3, 8)
    
    # Check available memory (at least 4GB recommended)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        requirements['memory'] = memory_gb >= 4.0
    except ImportError:
        requirements['memory'] = True  # Assume OK if psutil not available
    
    # Check disk space (at least 2GB free)
    try:
        disk_usage = os.statvfs('.')
        free_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
        requirements['disk_space'] = free_gb >= 2.0
    except AttributeError:
        # Windows doesn't have statvfs
        requirements['disk_space'] = True
    
    # Check pip availability
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        requirements['pip'] = True
    except subprocess.CalledProcessError:
        requirements['pip'] = False
    
    return requirements

def install_system_dependencies():
    """Install system-level dependencies if possible"""
    system = platform.system().lower()
    
    if console:
        console.print("[cyan]🔧 Checking system dependencies...[/cyan]")
    
    # Try to install TA-Lib system dependency
    if system == 'linux':
        try:
            if console:
                console.print("[yellow]📦 Installing TA-Lib system dependency on Linux...[/yellow]")
            subprocess.run(['apt-get', 'update'], check=False, capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', 'libta-lib-dev'], check=False, capture_output=True)
        except Exception:
            pass
    elif system == 'darwin':  # macOS
        try:
            if console:
                console.print("[yellow]📦 Installing TA-Lib system dependency on macOS...[/yellow]")
            subprocess.run(['brew', 'install', 'ta-lib'], check=False, capture_output=True)
        except Exception:
            pass

def get_requirements_from_file(filepath: str) -> List[str]:
    """Parse requirements file"""
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        dependencies = []
        for line in f:
            line = line.strip()
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if line and not line.startswith('#'):
                dependencies.append(line)
    return dependencies

def install_package_with_fallback(package: str, console_ref=None) -> bool:
    """Install package with fallback strategies"""
    strategies = [
        [sys.executable, '-m', 'pip', 'install', package],
        [sys.executable, '-m', 'pip', 'install', package, '--user'],
        [sys.executable, '-m', 'pip', 'install', package, '--no-cache-dir'],
        [sys.executable, '-m', 'pip', 'install', package, '--force-reinstall'],
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            if console_ref:
                strategy_name = ['standard', 'user', 'no-cache', 'force-reinstall'][i]
                console_ref.print(f"[dim]Trying {strategy_name} install for {package}...[/dim]")
            
            result = subprocess.run(strategy, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            continue
    
    return False

def create_missing_core_modules():
    """Create missing core modules with basic implementations"""
    core_dir = Path('core')
    
    missing_modules = [
        ('unified_config_manager.py', '''"""
🔧 Unified Configuration Manager
Placeholder implementation for NICEGOLD ProjectP
"""

class UnifiedConfigManager:
    """Basic configuration manager"""
    
    def __init__(self):
        self.config = {}
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value

_instance = None

def get_unified_config_manager():
    global _instance
    if _instance is None:
        _instance = UnifiedConfigManager()
    return _instance
'''),
        
        ('unified_resource_manager.py', '''"""
⚡ Unified Resource Manager  
Placeholder implementation for NICEGOLD ProjectP
"""
import psutil
import threading

class UnifiedResourceManager:
    """Basic resource manager"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.stats = {}
    
    def get_memory_usage(self):
        """Get current memory usage"""
        return psutil.virtual_memory().percent
    
    def get_cpu_usage(self):
        """Get current CPU usage"""
        return psutil.cpu_percent()
    
    def optimize_memory(self):
        """Placeholder memory optimization"""
        import gc
        gc.collect()

_instance = None

def get_unified_resource_manager():
    global _instance
    if _instance is None:
        _instance = UnifiedResourceManager()
    return _instance
'''),
        
        ('output_manager.py', '''"""
📄 Output Manager
Placeholder implementation for NICEGOLD ProjectP
"""
from pathlib import Path
import json

class NicegoldOutputManager:
    """Basic output manager"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_json(self, data, filename):
        """Save data as JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath
    
    def save_text(self, text, filename):
        """Save text to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(text)
        return filepath

_instance = None

def get_output_manager():
    global _instance
    if _instance is None:
        _instance = NicegoldOutputManager()
    return _instance
''')
    ]
    
    created = []
    for filename, content in missing_modules:
        filepath = core_dir / filename
        if not filepath.exists():
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                created.append(filename)
            except Exception as e:
                if console:
                    console.print(f"[red]❌ Could not create {filename}: {e}[/red]")
    
    return created

def main():
    """Main enterprise installation function"""
    
    if console:
        console.print(Panel.fit(
            "[bold cyan]🏢 NICEGOLD Enterprise ProjectP Installer[/bold cyan]",
            subtitle="[yellow]Advanced Enterprise Installation System[/yellow]"
        ))
    else:
        print("🏢 NICEGOLD Enterprise ProjectP Installer")
        print("=" * 60)
    
    # Check system requirements
    if console:
        console.print("[cyan]🔍 Checking system requirements...[/cyan]")
    
    system_reqs = check_system_requirements()
    
    if console:
        req_table = Table(title="System Requirements Check")
        req_table.add_column("Requirement", style="cyan")
        req_table.add_column("Status", style="white")
        req_table.add_column("Details", style="dim")
        
        for req, status in system_reqs.items():
            req_name = req.replace('_', ' ').title()
            status_text = "✅ OK" if status else "❌ FAIL"
            
            if req == 'python_version':
                details = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            elif req == 'memory':
                try:
                    import psutil
                    details = f"{psutil.virtual_memory().total / (1024**3):.1f} GB available"
                except:
                    details = "Unknown"
            elif req == 'disk_space':
                details = "Sufficient space"
            else:
                details = "Available"
            
            req_table.add_row(req_name, status_text, details)
        
        console.print(req_table)
    
    # Check if we can proceed
    if not all(system_reqs.values()):
        if console:
            console.print("[red]❌ System requirements not met. Please address the issues above.[/red]")
        else:
            print("❌ System requirements not met.")
        return False
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create missing core modules
    if console:
        console.print("[cyan]🔧 Creating missing core modules...[/cyan]")
    
    created_modules = create_missing_core_modules()
    if created_modules:
        if console:
            console.print(f"[green]✅ Created {len(created_modules)} core modules: {', '.join(created_modules)}[/green]")
    
    # Determine which requirements file to use
    requirements_files = ['requirements_complete.txt', 'requirements.txt']
    requirements_file = None
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            requirements_file = req_file
            break
    
    if not requirements_file:
        if console:
            console.print("[red]❌ No requirements file found![/red]")
        return False
    
    if console:
        console.print(f"[cyan]📦 Using requirements file: {requirements_file}[/cyan]")
    
    # Get dependencies
    dependencies = get_requirements_from_file(requirements_file)
    
    if not dependencies:
        if console:
            console.print("[yellow]⚠️ No dependencies found in requirements file.[/yellow]")
        return True
    
    if console:
        console.print(f"[cyan]Found {len(dependencies)} packages to install[/cyan]")
    
    # Install packages
    successful = []
    failed = []
    
    if console:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("[cyan]Installing packages...", total=len(dependencies))
            
            for dep in dependencies:
                progress.update(task, description=f"Installing {dep}...")
                
                if install_package_with_fallback(dep, console):
                    successful.append(dep)
                else:
                    failed.append(dep)
                
                progress.advance(task)
    else:
        for i, dep in enumerate(dependencies, 1):
            print(f"Installing {dep}... ({i}/{len(dependencies)})")
            if install_package_with_fallback(dep):
                successful.append(dep)
            else:
                failed.append(dep)
    
    # Summary
    if console:
        console.print("\n" + "="*60)
        summary_table = Table(title="Installation Summary")
        summary_table.add_column("Status", style="cyan")
        summary_table.add_column("Count", style="white")
        summary_table.add_column("Percentage", style="dim")
        
        total = len(dependencies)
        success_pct = (len(successful) / total * 100) if total > 0 else 0
        fail_pct = (len(failed) / total * 100) if total > 0 else 0
        
        summary_table.add_row("✅ Successful", str(len(successful)), f"{success_pct:.1f}%")
        summary_table.add_row("❌ Failed", str(len(failed)), f"{fail_pct:.1f}%")
        
        console.print(summary_table)
    else:
        print("\n" + "="*40)
        print("INSTALLATION SUMMARY")
        print("="*40)
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
    
    if failed:
        if console:
            console.print(f"\n[yellow]⚠️ {len(failed)} packages failed to install:[/yellow]")
            for pkg in failed[:10]:  # Show first 10
                console.print(f"  - [red]{pkg}[/red]")
            if len(failed) > 10:
                console.print(f"  ... and {len(failed) - 10} more")
        else:
            print(f"\n⚠️ {len(failed)} packages failed to install:")
            for pkg in failed[:10]:
                print(f"  - {pkg}")
    
    # Final status
    if len(failed) == 0:
        if console:
            console.print("\n[bold green]🎉 Enterprise installation completed successfully![/bold green]")
            console.print("[cyan]ℹ️ Run 'python check_installation.py' to verify the installation.[/cyan]")
        else:
            print("\n🎉 Enterprise installation completed successfully!")
        return True
    else:
        if console:
            console.print(f"\n[yellow]⚠️ Installation completed with {len(failed)} issues.[/yellow]")
            console.print("[cyan]ℹ️ You may need to install failed packages manually.[/cyan]")
        else:
            print(f"\n⚠️ Installation completed with {len(failed)} issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
