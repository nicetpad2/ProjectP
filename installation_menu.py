#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - INSTALLATION MENU
🚀 Interactive Installation System for Complete Dependency Management

This script provides an interactive menu for installing all required dependencies
for the NICEGOLD ProjectP enterprise system with beautiful progress tracking.

Author: NICEGOLD Enterprise ProjectP Team
Version: v3.0 Enterprise Edition
Date: July 12, 2025
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📦 Installing rich for better UI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=False)
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.prompt import Prompt, Confirm
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

class NicegoldInstaller:
    """
    🏢 NICEGOLD Enterprise ProjectP Installation System
    Interactive installer with beautiful UI and comprehensive dependency management
    """
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements_complete.txt"
        self.requirements_basic = self.project_root / "requirements.txt"
        
    def print_banner(self):
        """Display beautiful banner"""
        if RICH_AVAILABLE:
            banner = Text()
            banner.append("🏢 NICEGOLD ENTERPRISE PROJECTP\n", style="bold cyan")
            banner.append("AI-Powered Algorithmic Trading System\n", style="bold white")
            banner.append("🚀 Interactive Installation Menu", style="bold green")
            
            panel = Panel(
                banner,
                title="[bold blue]Installation System[/bold blue]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print("🏢 NICEGOLD ENTERPRISE PROJECTP")
            print("AI-Powered Algorithmic Trading System")
            print("🚀 Interactive Installation Menu")
            print("=" * 50)
    
    def show_menu(self):
        """Display main menu"""
        if RICH_AVAILABLE:
            table = Table(title="[bold cyan]Installation Options[/bold cyan]")
            table.add_column("Option", style="cyan", width=10)
            table.add_column("Description", style="white")
            table.add_column("Recommended", style="green")
            
            table.add_row("1", "🚀 Complete Enterprise Installation", "✅ YES")
            table.add_row("2", "📦 Basic Installation", "⚠️ Minimal")
            table.add_row("3", "🔧 Dependency Check & Fix", "🛠️ Repair")
            table.add_row("4", "🧪 System Requirements Check", "🔍 Verify")
            table.add_row("5", "📊 Package Status Report", "📋 Status")
            table.add_row("6", "🎯 Custom Installation", "⚙️ Advanced")
            table.add_row("Q", "🚪 Quit", "")
            
            self.console.print(table)
        else:
            print("\n📋 Installation Options:")
            print("1. 🚀 Complete Enterprise Installation (Recommended)")
            print("2. 📦 Basic Installation")
            print("3. 🔧 Dependency Check & Fix")
            print("4. 🧪 System Requirements Check")
            print("5. 📊 Package Status Report")
            print("6. 🎯 Custom Installation")
            print("Q. 🚪 Quit")
    
    def install_requirements(self, requirements_file, description="packages"):
        """Install requirements with progress tracking"""
        if not requirements_file.exists():
            self._print_error(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        self._print_info(f"📦 Installing {description}...")
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Installing {description}...", total=100)
                
                process = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Simulate progress (since pip doesn't provide real progress)
                for i in range(100):
                    time.sleep(0.1)
                    progress.update(task, advance=1)
                    if process.poll() is not None:
                        break
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    progress.update(task, completed=100)
                    self._print_success(f"✅ {description} installed successfully!")
                    return True
                else:
                    self._print_error(f"❌ Installation failed for {description}")
                    if stderr:
                        self._print_error(f"Error: {stderr}")
                    return False
        else:
            # Fallback without rich
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {description} installed successfully!")
                return True
            else:
                print(f"❌ Installation failed for {description}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
    
    def check_system_requirements(self):
        """Check system requirements"""
        self._print_info("🧪 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self._print_success(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self._print_error(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
            return False
        
        # Check platform
        platform_info = platform.platform()
        self._print_info(f"🖥️ Platform: {platform_info}")
        
        # Check pip
        try:
            import pip
            self._print_success(f"✅ pip available")
        except ImportError:
            self._print_error("❌ pip not available")
            return False
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 4:
                self._print_success(f"✅ Memory: {memory_gb:.1f} GB")
            else:
                self._print_warning(f"⚠️ Memory: {memory_gb:.1f} GB (recommended: 8+ GB)")
        except ImportError:
            self._print_warning("⚠️ Cannot check memory (psutil not available)")
        
        return True
    
    def check_package_status(self):
        """Check status of critical packages"""
        critical_packages = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
            'shap', 'optuna', 'rich', 'colorama', 'psutil', 'joblib'
        ]
        
        if RICH_AVAILABLE:
            table = Table(title="[bold cyan]Package Status[/bold cyan]")
            table.add_column("Package", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Version", style="yellow")
            
            for package in critical_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                    table.add_row(package, "[green]✅ Installed[/green]", version)
                except ImportError:
                    table.add_row(package, "[red]❌ Missing[/red]", "N/A")
            
            self.console.print(table)
        else:
            print("\n📊 Package Status:")
            for package in critical_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                    print(f"✅ {package}: {version}")
                except ImportError:
                    print(f"❌ {package}: Not installed")
    
    def fix_dependencies(self):
        """Fix common dependency issues"""
        self._print_info("🔧 Fixing common dependency issues...")
        
        # Update pip
        self._print_info("📦 Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Fix numpy/SHAP compatibility
        self._print_info("🔧 Fixing NumPy/SHAP compatibility...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
        
        # Install critical packages individually
        critical_packages = [
            "rich", "colorama", "psutil", "joblib", "PyYAML",
            "pandas", "scikit-learn", "shap", "optuna"
        ]
        
        for package in critical_packages:
            self._print_info(f"📦 Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True
            )
            if result.returncode == 0:
                self._print_success(f"✅ {package} installed")
            else:
                self._print_error(f"❌ Failed to install {package}")
    
    def custom_installation(self):
        """Custom installation with options"""
        if RICH_AVAILABLE:
            self.console.print("\n[bold cyan]🎯 Custom Installation Options[/bold cyan]")
            
            options = {
                "ml": "🧠 Machine Learning packages (TensorFlow, PyTorch)",
                "analysis": "📊 Data Analysis packages (pandas, numpy, scipy)",
                "viz": "📈 Visualization packages (matplotlib, plotly, seaborn)",
                "enterprise": "🏢 Enterprise packages (rich, colorama, psutil)",
                "optional": "🔧 Optional packages (cloud, monitoring)"
            }
            
            for key, description in options.items():
                if Confirm.ask(f"Install {description}?"):
                    self._install_category(key)
        else:
            print("\n🎯 Custom Installation")
            print("1. Machine Learning packages")
            print("2. Data Analysis packages") 
            print("3. Visualization packages")
            print("4. Enterprise packages")
            print("5. Optional packages")
            
            choice = input("Select categories (1-5, comma separated): ")
            categories = choice.split(',')
            for cat in categories:
                self._install_category(cat.strip())
    
    def _install_category(self, category):
        """Install specific category of packages"""
        packages = {
            "ml": ["tensorflow>=2.10.0", "torch>=1.12.0", "scikit-learn>=1.1.0"],
            "analysis": ["pandas>=2.0.0", "numpy>=1.21.0", "scipy>=1.8.0"],
            "viz": ["matplotlib>=3.5.0", "plotly>=6.2.0", "seaborn>=0.11.0"],
            "enterprise": ["rich>=12.0.0", "colorama>=0.4.4", "psutil>=5.8.0"],
            "optional": ["boto3", "google-cloud-storage", "azure-storage-blob"]
        }
        
        if category in packages:
            for package in packages[category]:
                subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    def _print_success(self, message):
        """Print success message"""
        if RICH_AVAILABLE:
            self.console.print(message, style="bold green")
        else:
            print(message)
    
    def _print_error(self, message):
        """Print error message"""
        if RICH_AVAILABLE:
            self.console.print(message, style="bold red")
        else:
            print(message)
    
    def _print_warning(self, message):
        """Print warning message"""
        if RICH_AVAILABLE:
            self.console.print(message, style="bold yellow")
        else:
            print(message)
    
    def _print_info(self, message):
        """Print info message"""
        if RICH_AVAILABLE:
            self.console.print(message, style="bold blue")
        else:
            print(message)
    
    def run(self):
        """Run the installation menu"""
        while True:
            self.print_banner()
            self.show_menu()
            
            if RICH_AVAILABLE:
                choice = Prompt.ask("\n🎯 Select an option", default="1")
            else:
                choice = input("\n🎯 Select an option (default: 1): ") or "1"
            
            if choice.lower() == 'q':
                self._print_info("👋 Goodbye!")
                break
            elif choice == '1':
                # Complete Enterprise Installation
                self._print_info("🚀 Starting Complete Enterprise Installation...")
                if self.check_system_requirements():
                    success = self.install_requirements(
                        self.requirements_file, 
                        "enterprise packages"
                    )
                    if success:
                        self._print_success("🎉 Complete Enterprise Installation finished!")
                    else:
                        self._print_error("❌ Installation completed with errors")
            elif choice == '2':
                # Basic Installation
                self._print_info("📦 Starting Basic Installation...")
                success = self.install_requirements(
                    self.requirements_basic,
                    "basic packages"
                )
                if success:
                    self._print_success("✅ Basic Installation finished!")
            elif choice == '3':
                # Dependency Check & Fix
                self.fix_dependencies()
                self._print_success("🔧 Dependency fixing completed!")
            elif choice == '4':
                # System Requirements Check
                self.check_system_requirements()
            elif choice == '5':
                # Package Status Report
                self.check_package_status()
            elif choice == '6':
                # Custom Installation
                self.custom_installation()
            else:
                self._print_error("❌ Invalid option. Please try again.")
            
            if RICH_AVAILABLE:
                if not Confirm.ask("\n🔄 Continue with installation menu?"):
                    break
            else:
                cont = input("\n🔄 Continue? (y/N): ")
                if cont.lower() != 'y':
                    break


if __name__ == "__main__":
    try:
        installer = NicegoldInstaller()
        installer.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)
