#!/usr/bin/env python3
"""
🔍 NICEGOLD ProjectP - Installation Checker
Complete dependency verification and system health check
"""

import subprocess
import sys
import os
import importlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set UTF-8 encoding for cross-platform compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

def install_rich_if_missing():
    """Install rich library if missing for better UI"""
    try:
        import rich
        from rich.console import Console
        return Console()
    except ImportError:
        print("📦 Installing 'rich' library for better display...")
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
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    
    def print_success(message: str):
        console.print(f"[green]✅ {message}[/green]")
    
    def print_error(message: str):
        console.print(f"[red]❌ {message}[/red]")
    
    def print_warning(message: str):
        console.print(f"[yellow]⚠️ {message}[/yellow]")
    
    def print_info(message: str):
        console.print(f"[cyan]ℹ️ {message}[/cyan]")
else:
    def print_success(message: str):
        print(f"✅ {message}")
    
    def print_error(message: str):
        print(f"❌ {message}")
    
    def print_warning(message: str):
        print(f"⚠️ {message}")
    
    def print_info(message: str):
        print(f"ℹ️ {message}")

def check_python_version() -> bool:
    """Check if Python version is supported"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} (supported)")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_pip_available() -> bool:
    """Check if pip is available"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print_success("pip is available")
        return True
    except subprocess.CalledProcessError:
        print_error("pip is not available")
        return False

def get_requirements_from_file(filepath: str) -> List[str]:
    """Parse requirements file and return clean package names"""
    if not os.path.exists(filepath):
        return []
    
    packages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove comments and extract package name
                package = line.split('#')[0].strip()
                if package:
                    # Extract just the package name (remove version constraints)
                    package_name = package.split('>=')[0].split('==')[0].split('<=')[0].split('>')[0].split('<')[0]
                    packages.append(package_name)
    return packages

def check_package_installed(package: str) -> bool:
    """Check if a Python package is installed"""
    try:
        # Handle special cases
        if package == 'opencv-python':
            importlib.import_module('cv2')
        elif package == 'Pillow':
            importlib.import_module('PIL')
        elif package == 'beautifulsoup4':
            importlib.import_module('bs4')
        elif package == 'scikit-learn':
            importlib.import_module('sklearn')
        elif package == 'python-dateutil':
            importlib.import_module('dateutil')
        elif package == 'PyYAML':
            importlib.import_module('yaml')
        elif package == 'google-colab':
            try:
                importlib.import_module('google.colab')
            except ImportError:
                # google-colab is only available in Colab environment
                return True
        else:
            importlib.import_module(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def check_core_modules() -> Tuple[List[str], List[str]]:
    """Check NICEGOLD core modules availability"""
    core_modules = [
        'core.unified_enterprise_logger',
        'core.project_paths',
        'core.enterprise_model_manager',
        'core.unified_config_manager',
        'core.unified_resource_manager',
        'core.output_manager'
    ]
    
    available = []
    missing = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            available.append(module)
        except ImportError:
            missing.append(module)
    
    return available, missing

def check_elliott_wave_modules() -> Tuple[List[str], List[str]]:
    """Check Elliott Wave modules availability"""
    elliott_modules = [
        'elliott_wave_modules.feature_selector',
        'elliott_wave_modules.data_processor',
        'elliott_wave_modules.cnn_lstm_engine',
        'elliott_wave_modules.dqn_agent',
        'elliott_wave_modules.pipeline_orchestrator',
        'elliott_wave_modules.performance_analyzer'
    ]
    
    available = []
    missing = []
    
    for module in elliott_modules:
        try:
            importlib.import_module(module)
            available.append(module)
        except ImportError:
            missing.append(module)
    
    return available, missing

def check_system_dependencies() -> Dict[str, bool]:
    """Check system-level dependencies"""
    checks = {}
    
    # Check for TA-Lib system dependency
    try:
        import talib
        checks['TA-Lib'] = True
    except ImportError:
        checks['TA-Lib'] = False
    
    # Check for Git (optional)
    try:
        subprocess.check_call(['git', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        checks['Git'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks['Git'] = False
    
    return checks

def main():
    """Main installation check function"""
    if console:
        console.print(Panel.fit(
            "[bold cyan]🔍 NICEGOLD ProjectP Installation Checker[/bold cyan]",
            subtitle="[yellow]Verifying system readiness[/yellow]"
        ))
    else:
        print("🔍 NICEGOLD ProjectP Installation Checker")
        print("=" * 50)
    
    print_info("Starting comprehensive installation check...")
    print("")
    
    # Check Python version
    print_info("Checking Python version...")
    python_ok = check_python_version()
    
    # Check pip
    print_info("Checking pip availability...")
    pip_ok = check_pip_available()
    
    # Check packages from requirements files
    print_info("Checking installed packages...")
    
    # Check main requirements
    main_requirements = get_requirements_from_file('requirements.txt')
    complete_requirements = get_requirements_from_file('requirements_complete.txt')
    
    # Use complete requirements if available, otherwise main requirements
    requirements = complete_requirements if complete_requirements else main_requirements
    
    if not requirements:
        print_warning("No requirements file found!")
        return
    
    print_info(f"Found {len(requirements)} packages to check")
    
    installed_packages = []
    missing_packages = []
    
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Checking packages...", total=len(requirements))
            
            for package in requirements:
                progress.update(task, description=f"Checking {package}...")
                if check_package_installed(package):
                    installed_packages.append(package)
                else:
                    missing_packages.append(package)
                progress.advance(task)
    else:
        for i, package in enumerate(requirements, 1):
            print(f"Checking {package}... ({i}/{len(requirements)})")
            if check_package_installed(package):
                installed_packages.append(package)
            else:
                missing_packages.append(package)
    
    # Check core modules
    print_info("Checking NICEGOLD core modules...")
    available_core, missing_core = check_core_modules()
    
    # Check Elliott Wave modules
    print_info("Checking Elliott Wave modules...")
    available_elliott, missing_elliott = check_elliott_wave_modules()
    
    # Check system dependencies
    print_info("Checking system dependencies...")
    system_deps = check_system_dependencies()
    
    # Print summary
    print("")
    if console:
        # Create summary table
        table = Table(title="Installation Summary", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")
        
        # Python & pip
        table.add_row("Python", "✅ OK" if python_ok else "❌ FAIL", 
                     f"Version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        table.add_row("pip", "✅ OK" if pip_ok else "❌ FAIL", "Package manager")
        
        # Packages
        table.add_row("Python Packages", 
                     f"✅ {len(installed_packages)}/{len(requirements)}" if len(missing_packages) == 0 else f"⚠️ {len(installed_packages)}/{len(requirements)}", 
                     f"{len(missing_packages)} missing" if missing_packages else "All installed")
        
        # Core modules
        table.add_row("Core Modules", 
                     f"✅ {len(available_core)}/{len(available_core + missing_core)}" if len(missing_core) == 0 else f"⚠️ {len(available_core)}/{len(available_core + missing_core)}", 
                     f"{len(missing_core)} missing" if missing_core else "All available")
        
        # Elliott Wave modules
        table.add_row("Elliott Wave Modules", 
                     f"✅ {len(available_elliott)}/{len(available_elliott + missing_elliott)}" if len(missing_elliott) == 0 else f"⚠️ {len(available_elliott)}/{len(available_elliott + missing_elliott)}", 
                     f"{len(missing_elliott)} missing" if missing_elliott else "All available")
        
        console.print(table)
    else:
        print("=" * 50)
        print("INSTALLATION SUMMARY")
        print("=" * 50)
        print(f"Python: {'✅ OK' if python_ok else '❌ FAIL'}")
        print(f"pip: {'✅ OK' if pip_ok else '❌ FAIL'}")
        print(f"Packages: {len(installed_packages)}/{len(requirements)} installed")
        print(f"Core Modules: {len(available_core)}/{len(available_core + missing_core)} available")
        print(f"Elliott Wave Modules: {len(available_elliott)}/{len(available_elliott + missing_elliott)} available")
    
    # Print missing items
    if missing_packages:
        print("")
        print_warning(f"Missing {len(missing_packages)} packages:")
        for package in missing_packages[:10]:  # Show first 10
            print(f"  - {package}")
        if len(missing_packages) > 10:
            print(f"  ... and {len(missing_packages) - 10} more")
    
    if missing_core:
        print("")
        print_warning(f"Missing {len(missing_core)} core modules:")
        for module in missing_core:
            print(f"  - {module}")
    
    if missing_elliott:
        print("")
        print_warning(f"Missing {len(missing_elliott)} Elliott Wave modules:")
        for module in missing_elliott:
            print(f"  - {module}")
    
    # Print system dependencies
    print("")
    print_info("System Dependencies:")
    for dep, available in system_deps.items():
        if available:
            print_success(f"{dep} is available")
        else:
            print_warning(f"{dep} is not available (optional)")
    
    # Final verdict
    print("")
    total_issues = len(missing_packages) + len(missing_core) + len(missing_elliott)
    if total_issues == 0:
        print_success("🎉 Installation is complete! All dependencies are satisfied.")
        print_info("You can now run the main application: python ProjectP.py")
    else:
        print_warning(f"⚠️ Installation has {total_issues} issues that need attention.")
        print_info("Run the installer to fix missing dependencies: python install_dependencies.py")

if __name__ == "__main__":
    main()
