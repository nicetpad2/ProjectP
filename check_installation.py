#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - INSTALLATION VERIFICATION
🔍 Comprehensive Installation Check and Validation System

This script verifies that all required dependencies are properly installed
and the NICEGOLD ProjectP system is ready for production use.

Author: NICEGOLD Enterprise ProjectP Team
Version: v3.0 Enterprise Edition
Date: July 12, 2025
"""

import sys
import os
import importlib
import platform
import subprocess
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class InstallationChecker:
    """
    🔍 NICEGOLD Enterprise ProjectP Installation Verification System
    Comprehensive verification of all dependencies and system readiness
    """
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.project_root = Path(__file__).parent
        
        # Critical packages for NICEGOLD ProjectP
        self.critical_packages = {
            # Core Data Science
            'numpy': '1.21.0',
            'pandas': '2.0.0',
            'scipy': '1.8.0',
            'scikit-learn': '1.1.0',
            'joblib': '1.1.0',
            
            # Machine Learning
            'tensorflow': '2.10.0',
            'torch': '1.12.0',
            
            # Feature Selection (CRITICAL)
            'shap': '0.41.0',
            'optuna': '3.0.0',
            
            # Enterprise UI
            'rich': '12.0.0',
            'colorama': '0.4.4',
            'psutil': '5.8.0',
            
            # Configuration
            'yaml': None  # PyYAML
        }
        
        # Optional but recommended packages
        self.optional_packages = {
            'xgboost': '1.6.0',
            'lightgbm': '3.3.0',
            'gymnasium': '0.26.0',
            'stable_baselines3': '1.6.0',
            'yfinance': '0.1.70',
            'ta': '0.10.0',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'plotly': '6.2.0'
        }
        
        # Enterprise specific packages
        self.enterprise_packages = {
            'alembic': '1.8.0',
            'cryptography': '3.4.0',
            'sqlalchemy': '1.4.0',
            'h5py': '3.7.0'
        }
        
        self.results = {
            'critical': {},
            'optional': {},
            'enterprise': {},
            'system': {}
        }
    
    def print_banner(self):
        """Display verification banner"""
        if RICH_AVAILABLE:
            banner = Text()
            banner.append("🔍 NICEGOLD ENTERPRISE PROJECTP\n", style="bold cyan")
            banner.append("Installation Verification System\n", style="bold white")
            banner.append("🚀 Checking System Readiness", style="bold green")
            
            panel = Panel(
                banner,
                title="[bold blue]Installation Check[/bold blue]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print("🔍 NICEGOLD ENTERPRISE PROJECTP")
            print("Installation Verification System")
            print("🚀 Checking System Readiness")
            print("=" * 50)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        
        if version >= (3, 8):
            status = "✅ Compatible"
            success = True
        elif version >= (3, 7):
            status = "⚠️ Minimum (upgrade recommended)"
            success = True
        else:
            status = "❌ Incompatible (requires 3.8+)"
            success = False
        
        self.results['system']['python'] = {
            'version': f"{version.major}.{version.minor}.{version.micro}",
            'status': status,
            'success': success
        }
        
        return success
    
    def check_platform_info(self):
        """Check platform information"""
        system = platform.system()
        machine = platform.machine()
        platform_info = platform.platform()
        
        self.results['system']['platform'] = {
            'system': system,
            'machine': machine,
            'platform': platform_info,
            'status': "✅ Detected",
            'success': True
        }
        
        return True
    
    def check_memory(self):
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 8:
                status = "✅ Excellent"
                success = True
            elif memory_gb >= 4:
                status = "⚠️ Adequate"
                success = True
            else:
                status = "❌ Insufficient (4GB+ recommended)"
                success = False
            
            self.results['system']['memory'] = {
                'total_gb': f"{memory_gb:.1f}",
                'available_gb': f"{memory.available / (1024**3):.1f}",
                'status': status,
                'success': success
            }
            
            return success
        except ImportError:
            self.results['system']['memory'] = {
                'status': "⚠️ Cannot check (psutil not available)",
                'success': True
            }
            return True
    
    def check_package(self, package_name, min_version=None, category='critical'):
        """Check if a package is installed and meets version requirements"""
        try:
            # Special handling for yaml (PyYAML) and scikit-learn
            if package_name == 'yaml':
                import yaml
                module = yaml
                actual_package = 'PyYAML'
            elif package_name == 'scikit-learn':
                import sklearn
                module = sklearn
                actual_package = 'scikit-learn'
            else:
                module = importlib.import_module(package_name)
                actual_package = package_name
            
            # Get version
            version = getattr(module, '__version__', 'Unknown')
            
            # Check version compatibility
            if min_version and version != 'Unknown':
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) >= pkg_version.parse(min_version):
                        status = f"✅ {version}"
                        success = True
                    else:
                        status = f"⚠️ {version} (min: {min_version})"
                        success = False
                except:
                    # Fallback to string comparison
                    status = f"✅ {version}"
                    success = True
            else:
                status = f"✅ {version}"
                success = True
            
            self.results[category][package_name] = {
                'installed': True,
                'version': version,
                'status': status,
                'success': success,
                'package': actual_package
            }
            
            return success
            
        except ImportError:
            self.results[category][package_name] = {
                'installed': False,
                'version': 'Not installed',
                'status': "❌ Missing",
                'success': False,
                'package': package_name
            }
            return False
    
    def check_project_files(self):
        """Check if essential project files exist"""
        essential_files = [
            'ProjectP.py',
            'requirements.txt',
            'requirements_complete.txt',
            'core/menu_system.py',
            'elliott_wave_modules/data_processor.py',
            'menu_modules',
            'datacsv'
        ]
        
        missing_files = []
        
        for file_path in essential_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results['system']['project_files'] = {
                'status': f"❌ Missing: {', '.join(missing_files)}",
                'success': False
            }
            return False
        else:
            self.results['system']['project_files'] = {
                'status': "✅ All essential files present",
                'success': True
            }
            return True
    
    def run_verification(self):
        """Run complete verification process"""
        self.print_banner()
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                # System checks
                task1 = progress.add_task("🔍 Checking system requirements...", total=None)
                self.check_python_version()
                self.check_platform_info()
                self.check_memory()
                self.check_project_files()
                progress.update(task1, completed=True)
                
                # Critical packages
                task2 = progress.add_task("📦 Checking critical packages...", total=None)
                for package, min_version in self.critical_packages.items():
                    self.check_package(package, min_version, 'critical')
                progress.update(task2, completed=True)
                
                # Optional packages
                task3 = progress.add_task("🔧 Checking optional packages...", total=None)
                for package, min_version in self.optional_packages.items():
                    self.check_package(package, min_version, 'optional')
                progress.update(task3, completed=True)
                
                # Enterprise packages
                task4 = progress.add_task("🏢 Checking enterprise packages...", total=None)
                for package, min_version in self.enterprise_packages.items():
                    self.check_package(package, min_version, 'enterprise')
                progress.update(task4, completed=True)
        else:
            print("🔍 Checking system requirements...")
            self.check_python_version()
            self.check_platform_info()
            self.check_memory()
            self.check_project_files()
            
            print("📦 Checking critical packages...")
            for package, min_version in self.critical_packages.items():
                self.check_package(package, min_version, 'critical')
            
            print("🔧 Checking optional packages...")
            for package, min_version in self.optional_packages.items():
                self.check_package(package, min_version, 'optional')
            
            print("🏢 Checking enterprise packages...")
            for package, min_version in self.enterprise_packages.items():
                self.check_package(package, min_version, 'enterprise')
    
    def display_results(self):
        """Display verification results"""
        if RICH_AVAILABLE:
            # System Information Table
            sys_table = Table(title="[bold cyan]System Information[/bold cyan]")
            sys_table.add_column("Component", style="cyan")
            sys_table.add_column("Details", style="white")
            sys_table.add_column("Status", style="white")
            
            for component, info in self.results['system'].items():
                if component == 'python':
                    sys_table.add_row("Python Version", info['version'], info['status'])
                elif component == 'platform':
                    sys_table.add_row("Platform", info['platform'], info['status'])
                elif component == 'memory':
                    if 'total_gb' in info:
                        sys_table.add_row("Memory", f"{info['total_gb']} GB total", info['status'])
                    else:
                        sys_table.add_row("Memory", "Unknown", info['status'])
                elif component == 'project_files':
                    sys_table.add_row("Project Files", "Essential files", info['status'])
            
            self.console.print(sys_table)
            
            # Critical Packages Table
            crit_table = Table(title="[bold red]Critical Packages[/bold red]")
            crit_table.add_column("Package", style="cyan")
            crit_table.add_column("Version", style="yellow")
            crit_table.add_column("Status", style="white")
            
            for package, info in self.results['critical'].items():
                crit_table.add_row(package, info['version'], info['status'])
            
            self.console.print(crit_table)
            
            # Optional Packages Table (abbreviated)
            opt_table = Table(title="[bold yellow]Optional Packages[/bold yellow]")
            opt_table.add_column("Package", style="cyan")
            opt_table.add_column("Status", style="white")
            
            for package, info in self.results['optional'].items():
                status = "✅ Installed" if info['installed'] else "❌ Missing"
                opt_table.add_row(package, status)
            
            self.console.print(opt_table)
            
        else:
            # Fallback text display
            print("\n📊 VERIFICATION RESULTS")
            print("=" * 50)
            
            print("\n🖥️ System Information:")
            for component, info in self.results['system'].items():
                print(f"  {component}: {info['status']}")
            
            print("\n📦 Critical Packages:")
            for package, info in self.results['critical'].items():
                print(f"  {package}: {info['status']}")
            
            print("\n🔧 Optional Packages:")
            for package, info in self.results['optional'].items():
                status = "✅ Installed" if info['installed'] else "❌ Missing"
                print(f"  {package}: {status}")
    
    def generate_summary(self):
        """Generate verification summary"""
        # Count results
        critical_failed = sum(1 for info in self.results['critical'].values() if not info['success'])
        critical_total = len(self.results['critical'])
        
        optional_installed = sum(1 for info in self.results['optional'].values() if info['installed'])
        optional_total = len(self.results['optional'])
        
        system_failed = sum(1 for info in self.results['system'].values() if not info['success'])
        
        # Determine overall status
        if critical_failed == 0 and system_failed == 0:
            overall_status = "🎉 READY FOR PRODUCTION"
            status_color = "bold green"
        elif critical_failed == 0:
            overall_status = "✅ READY (with minor issues)"
            status_color = "bold yellow"
        else:
            overall_status = "❌ NOT READY (critical issues)"
            status_color = "bold red"
        
        if RICH_AVAILABLE:
            summary = Text()
            summary.append("📋 VERIFICATION SUMMARY\n\n", style="bold cyan")
            summary.append(f"Critical Packages: {critical_total - critical_failed}/{critical_total} ✅\n", 
                          style="white")
            summary.append(f"Optional Packages: {optional_installed}/{optional_total} ✅\n", 
                          style="white")
            summary.append(f"System Requirements: {'✅' if system_failed == 0 else '❌'}\n\n", 
                          style="white")
            summary.append(f"Overall Status: {overall_status}", style=status_color)
            
            panel = Panel(
                summary,
                title="[bold blue]Summary[/bold blue]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print("\n📋 VERIFICATION SUMMARY")
            print("=" * 30)
            print(f"Critical Packages: {critical_total - critical_failed}/{critical_total} ✅")
            print(f"Optional Packages: {optional_installed}/{optional_total} ✅")
            print(f"System Requirements: {'✅' if system_failed == 0 else '❌'}")
            print(f"\nOverall Status: {overall_status}")
        
        # Recommendations
        if critical_failed > 0:
            self._print_error("\n🚨 CRITICAL ISSUES FOUND:")
            self._print_error("Run: python installation_menu.py")
            self._print_error("Select option 1 (Complete Enterprise Installation)")
        
        if system_failed > 0:
            self._print_warning("\n⚠️ SYSTEM ISSUES FOUND:")
            self._print_warning("Check system requirements and resolve issues")
        
        return critical_failed == 0 and system_failed == 0
    
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
    
    def run(self):
        """Run complete installation check"""
        self.run_verification()
        self.display_results()
        success = self.generate_summary()
        
        return success


def main():
    """Main function"""
    try:
        checker = InstallationChecker()
        success = checker.run()
        
        if success:
            print("\n🚀 System is ready! Run: python ProjectP.py")
            sys.exit(0)
        else:
            print("\n❌ System not ready. Please resolve issues first.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
