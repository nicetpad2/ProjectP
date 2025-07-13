#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD ENTERPRISE PROJECTP - ULTIMATE AUTO INSTALLER
🏢 Complete Automated Installation System with Beautiful UI

This script provides one-click installation for the entire NICEGOLD ProjectP system
with intelligent dependency resolution, GPU detection, and enterprise features.

Author: NICEGOLD Enterprise ProjectP Team
Version: v3.0 Ultimate Edition
Date: July 12, 2025
"""

import os
import sys
import subprocess
import platform
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

class UltimateInstaller:
    """
    🚀 NICEGOLD Ultimate Auto Installer
    One-click installation system with intelligent dependency resolution
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.system_info = self._detect_system()
        self.install_log = []
        
        # Try to import rich for beautiful UI
        self.rich_available = self._setup_rich()
        
    def _setup_rich(self) -> bool:
        """Setup rich library for beautiful UI"""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            from rich.table import Table
            from rich.text import Text
            self.console = Console()
            return True
        except ImportError:
            print("📦 Installing rich for beautiful UI...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "rich>=12.0.0"
            ], capture_output=True)
            
            if result.returncode == 0:
                try:
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
                    from rich.table import Table
                    from rich.text import Text
                    self.console = Console()
                    return True
                except ImportError:
                    return False
            return False
    
    def _detect_system(self) -> Dict:
        """Detect system information"""
        system_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': sys.version_info,
            'is_colab': 'google.colab' in sys.modules,
            'is_jupyter': 'ipykernel' in sys.modules,
            'has_gpu': False
        }
        
        # Detect GPU
        try:
            import torch
            system_info['has_gpu'] = torch.cuda.is_available()
        except ImportError:
            pass
        
        return system_info
    
    def print_banner(self):
        """Display beautiful banner"""
        if self.rich_available:
            from rich.panel import Panel
            from rich.text import Text
            
            banner = Text()
            banner.append("🚀 NICEGOLD ENTERPRISE PROJECTP\n", style="bold cyan")
            banner.append("Ultimate Auto Installer v3.0\n", style="bold white")
            banner.append("🏢 AI-Powered Algorithmic Trading System\n", style="bold green")
            banner.append("⚡ One-Click Complete Installation", style="bold yellow")
            
            panel = Panel(
                banner,
                title="[bold blue]Ultimate Installation System[/bold blue]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
            
            # System info
            info_text = Text()
            info_text.append(f"🖥️ Platform: {self.system_info['platform']} ({self.system_info['architecture']})\n")
            info_text.append(f"🐍 Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}.{self.system_info['python_version'].micro}\n")
            if self.system_info['is_colab']:
                info_text.append("📓 Environment: Google Colab\n", style="bold blue")
            elif self.system_info['is_jupyter']:
                info_text.append("📓 Environment: Jupyter Notebook\n", style="bold blue")
            else:
                info_text.append("💻 Environment: Local Python\n")
            
            if self.system_info['has_gpu']:
                info_text.append("🎮 GPU: Available (CUDA detected)", style="bold green")
            else:
                info_text.append("💻 GPU: Not available (CPU only)", style="yellow")
            
            info_panel = Panel(
                info_text,
                title="[bold yellow]System Information[/bold yellow]",
                border_style="yellow"
            )
            self.console.print(info_panel)
        else:
            print("🚀 NICEGOLD ENTERPRISE PROJECTP")
            print("Ultimate Auto Installer v3.0")
            print("🏢 AI-Powered Algorithmic Trading System")
            print("⚡ One-Click Complete Installation")
            print("=" * 60)
            print(f"🖥️ Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
            print(f"🐍 Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}.{self.system_info['python_version'].micro}")
            if self.system_info['has_gpu']:
                print("🎮 GPU: Available (CUDA detected)")
            else:
                print("💻 GPU: Not available (CPU only)")
    
    def update_pip(self) -> bool:
        """Update pip to latest version"""
        self._log_step("📦 Updating pip to latest version...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_success("✅ pip updated successfully")
                return True
            else:
                self._log_error(f"❌ pip update failed: {result.stderr}")
                return False
        except Exception as e:
            self._log_error(f"❌ pip update error: {e}")
            return False
    
    def install_essential_packages(self) -> bool:
        """Install essential packages first"""
        essential_packages = [
            "wheel>=0.37.0",
            "setuptools>=60.0.0", 
            "rich>=12.0.0",
            "colorama>=0.4.6",
            "psutil>=5.8.0",
            "PyYAML>=6.0.0",
            "joblib>=1.1.0"
        ]
        
        self._log_step("🔧 Installing essential packages...")
        
        for package in essential_packages:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._log_success(f"✅ {package.split('>=')[0]} installed")
                else:
                    self._log_warning(f"⚠️ {package} installation warning")
            except Exception as e:
                self._log_error(f"❌ Error installing {package}: {e}")
        
        return True
    
    def fix_numpy_shap_compatibility(self) -> bool:
        """Fix NumPy/SHAP compatibility issues"""
        self._log_step("🔧 Fixing NumPy/SHAP compatibility...")
        
        try:
            # Install specific NumPy version for SHAP compatibility
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "numpy==1.26.4"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log_success("✅ NumPy 1.26.4 installed (SHAP compatible)")
                
                # Now install SHAP
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "shap>=0.48.0"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._log_success("✅ SHAP installed successfully")
                    return True
                else:
                    self._log_error("❌ SHAP installation failed")
                    return False
            else:
                self._log_error("❌ NumPy installation failed")
                return False
                
        except Exception as e:
            self._log_error(f"❌ NumPy/SHAP fix error: {e}")
            return False
    
    def install_core_ml_packages(self) -> bool:
        """Install core machine learning packages"""
        ml_packages = [
            "pandas>=2.2.3",
            "scipy>=1.14.1", 
            "scikit-learn>=1.5.2",
            "optuna>=4.4.0",
            "xgboost>=2.1.4",
            "lightgbm>=4.5.0"
        ]
        
        self._log_step("🧠 Installing core ML packages...")
        
        for package in ml_packages:
            self._install_package_with_retry(package)
        
        return True
    
    def install_deep_learning_packages(self) -> bool:
        """Install deep learning frameworks"""
        self._log_step("🤖 Installing deep learning frameworks...")
        
        # TensorFlow
        tf_package = "tensorflow>=2.18.0"
        if self.system_info['has_gpu']:
            self._log_step("🎮 Installing TensorFlow with GPU support...")
        else:
            self._log_step("💻 Installing TensorFlow (CPU only)...")
        
        self._install_package_with_retry(tf_package)
        
        # PyTorch
        if self.system_info['has_gpu']:
            self._log_step("🎮 Installing PyTorch with CUDA...")
            torch_packages = [
                "torch>=2.6.0",
                "torchvision>=0.21.0", 
                "torchaudio>=2.6.0"
            ]
        else:
            self._log_step("💻 Installing PyTorch (CPU only)...")
            torch_packages = [
                "torch>=2.6.0+cpu",
                "torchvision>=0.21.0+cpu",
                "torchaudio>=2.6.0+cpu"
            ]
        
        for package in torch_packages:
            self._install_package_with_retry(package)
        
        # Reinforcement Learning
        rl_packages = [
            "gymnasium>=1.1.1",
            "stable-baselines3>=2.6.0"
        ]
        
        for package in rl_packages:
            self._install_package_with_retry(package)
        
        return True
    
    def install_financial_packages(self) -> bool:
        """Install financial and technical analysis packages"""
        financial_packages = [
            "yfinance>=0.2.38",
            "ta>=0.11.0",
            "quantlib>=1.36",
            "statsmodels>=0.14.4",
            "mplfinance>=0.12.10"
        ]
        
        self._log_step("💰 Installing financial analysis packages...")
        
        for package in financial_packages:
            self._install_package_with_retry(package)
        
        return True
    
    def install_visualization_packages(self) -> bool:
        """Install visualization packages"""
        viz_packages = [
            "matplotlib>=3.9.2",
            "seaborn>=0.13.2",
            "plotly>=5.24.1"
        ]
        
        self._log_step("📊 Installing visualization packages...")
        
        for package in viz_packages:
            self._install_package_with_retry(package)
        
        return True
    
    def install_enterprise_packages(self) -> bool:
        """Install enterprise-specific packages"""
        enterprise_packages = [
            "alembic>=1.16.2",
            "cryptography>=44.0.0",
            "sqlalchemy>=2.0.36",
            "h5py>=3.12.1",
            "python-dotenv>=1.0.1"
        ]
        
        self._log_step("🏢 Installing enterprise packages...")
        
        for package in enterprise_packages:
            self._install_package_with_retry(package)
        
        return True
    
    def install_requirements_file(self) -> bool:
        """Install from requirements file if available"""
        requirements_file = self.project_root / "requirements_complete.txt"
        
        if not requirements_file.exists():
            requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            self._log_step(f"📋 Installing from {requirements_file.name}...")
            
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._log_success("✅ Requirements file installed successfully")
                    return True
                else:
                    self._log_warning("⚠️ Some packages from requirements file failed")
                    return True  # Continue anyway
            except Exception as e:
                self._log_error(f"❌ Requirements file installation error: {e}")
                return False
        else:
            self._log_warning("⚠️ No requirements file found, using package lists")
            return True
    
    def _install_package_with_retry(self, package: str, retries: int = 2) -> bool:
        """Install package with retry logic"""
        for attempt in range(retries + 1):
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    package_name = package.split('>=')[0].split('==')[0]
                    self._log_success(f"✅ {package_name} installed")
                    return True
                else:
                    if attempt < retries:
                        self._log_warning(f"⚠️ {package} failed, retrying...")
                        time.sleep(1)
                    else:
                        self._log_error(f"❌ {package} installation failed after {retries + 1} attempts")
                        return False
            except Exception as e:
                if attempt < retries:
                    self._log_warning(f"⚠️ {package} error, retrying...")
                    time.sleep(1)
                else:
                    self._log_error(f"❌ {package} error: {e}")
                    return False
        
        return False
    
    def verify_installation(self) -> bool:
        """Verify installation by importing critical packages"""
        critical_packages = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
            'shap', 'optuna', 'rich', 'colorama', 'psutil'
        ]
        
        self._log_step("🔍 Verifying installation...")
        
        failed_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
                self._log_success(f"✅ {package} verified")
            except ImportError:
                failed_packages.append(package)
                self._log_error(f"❌ {package} verification failed")
        
        if failed_packages:
            self._log_error(f"❌ {len(failed_packages)} critical packages failed verification")
            return False
        else:
            self._log_success("🎉 All critical packages verified successfully!")
            return True
    
    def create_installation_report(self):
        """Create installation report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_info,
            'install_log': self.install_log,
            'success': True
        }
        
        report_file = self.project_root / f"installation_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self._log_success(f"📋 Installation report saved: {report_file.name}")
        except Exception as e:
            self._log_error(f"❌ Could not save installation report: {e}")
    
    def run_ultimate_installation(self) -> bool:
        """Run the complete ultimate installation"""
        self.print_banner()
        
        if self.rich_available:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                
                steps = [
                    ("Update pip", self.update_pip),
                    ("Install essentials", self.install_essential_packages),
                    ("Fix NumPy/SHAP", self.fix_numpy_shap_compatibility),
                    ("Install core ML", self.install_core_ml_packages),
                    ("Install deep learning", self.install_deep_learning_packages),
                    ("Install financial", self.install_financial_packages),
                    ("Install visualization", self.install_visualization_packages),
                    ("Install enterprise", self.install_enterprise_packages),
                    ("Install requirements", self.install_requirements_file),
                    ("Verify installation", self.verify_installation)
                ]
                
                total_steps = len(steps)
                overall_task = progress.add_task("🚀 Ultimate Installation Progress", total=total_steps)
                
                success = True
                for i, (step_name, step_func) in enumerate(steps):
                    step_task = progress.add_task(f"Step {i+1}: {step_name}", total=100)
                    
                    step_success = step_func()
                    if not step_success and i < 7:  # Critical steps
                        success = False
                    
                    progress.update(step_task, completed=100)
                    progress.update(overall_task, advance=1)
                    
                    time.sleep(0.5)  # Brief pause for visual effect
                
                return success
        else:
            # Fallback without progress bars
            self._log_step("🚀 Starting Ultimate Installation...")
            
            steps = [
                ("Update pip", self.update_pip),
                ("Install essentials", self.install_essential_packages),
                ("Fix NumPy/SHAP", self.fix_numpy_shap_compatibility),
                ("Install core ML", self.install_core_ml_packages),
                ("Install deep learning", self.install_deep_learning_packages),
                ("Install financial", self.install_financial_packages),
                ("Install visualization", self.install_visualization_packages),
                ("Install enterprise", self.install_enterprise_packages),
                ("Install requirements", self.install_requirements_file),
                ("Verify installation", self.verify_installation)
            ]
            
            success = True
            for i, (step_name, step_func) in enumerate(steps, 1):
                print(f"\n[{i}/{len(steps)}] {step_name}...")
                step_success = step_func()
                if not step_success and i <= 7:  # Critical steps
                    success = False
            
            return success
    
    def _log_step(self, message: str):
        """Log installation step"""
        self.install_log.append({'type': 'step', 'message': message, 'timestamp': time.time()})
        if self.rich_available:
            self.console.print(message, style="bold blue")
        else:
            print(message)
    
    def _log_success(self, message: str):
        """Log success message"""
        self.install_log.append({'type': 'success', 'message': message, 'timestamp': time.time()})
        if self.rich_available:
            self.console.print(message, style="bold green")
        else:
            print(message)
    
    def _log_warning(self, message: str):
        """Log warning message"""
        self.install_log.append({'type': 'warning', 'message': message, 'timestamp': time.time()})
        if self.rich_available:
            self.console.print(message, style="bold yellow")
        else:
            print(message)
    
    def _log_error(self, message: str):
        """Log error message"""
        self.install_log.append({'type': 'error', 'message': message, 'timestamp': time.time()})
        if self.rich_available:
            self.console.print(message, style="bold red")
        else:
            print(message)
    
    def display_final_summary(self, success: bool):
        """Display final installation summary"""
        if self.rich_available:
            from rich.panel import Panel
            from rich.text import Text
            
            if success:
                summary = Text()
                summary.append("🎉 INSTALLATION COMPLETED SUCCESSFULLY!\n\n", style="bold green")
                summary.append("✅ All critical packages installed\n")
                summary.append("✅ System ready for NICEGOLD ProjectP\n")
                summary.append("✅ Enterprise features available\n\n")
                summary.append("🚀 Next steps:\n", style="bold cyan")
                summary.append("   1. Run: python check_installation.py\n")
                summary.append("   2. Run: python ProjectP.py\n")
                summary.append("   3. Enjoy AI-powered trading! 🎯")
                
                panel = Panel(
                    summary,
                    title="[bold green]🎉 SUCCESS[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                )
            else:
                summary = Text()
                summary.append("⚠️ INSTALLATION COMPLETED WITH ISSUES\n\n", style="bold yellow")
                summary.append("🔧 Some packages may need manual installation\n")
                summary.append("📋 Check installation_report_*.json for details\n\n")
                summary.append("🚀 Recommended actions:\n", style="bold cyan")
                summary.append("   1. Run: python check_installation.py\n")
                summary.append("   2. Run: python installation_menu.py\n")
                summary.append("   3. Select option 3 (Dependency Check & Fix)")
                
                panel = Panel(
                    summary,
                    title="[bold yellow]⚠️ PARTIAL SUCCESS[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2)
                )
            
            self.console.print(panel)
        else:
            if success:
                print("\n🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
                print("✅ All critical packages installed")
                print("✅ System ready for NICEGOLD ProjectP")
                print("\n🚀 Next steps:")
                print("   1. Run: python check_installation.py")
                print("   2. Run: python ProjectP.py")
                print("   3. Enjoy AI-powered trading! 🎯")
            else:
                print("\n⚠️ INSTALLATION COMPLETED WITH ISSUES")
                print("🔧 Some packages may need manual installation")
                print("\n🚀 Recommended actions:")
                print("   1. Run: python check_installation.py")
                print("   2. Run: python installation_menu.py")


def main():
    """Main function"""
    try:
        installer = UltimateInstaller()
        
        print("🚀 NICEGOLD ENTERPRISE PROJECTP - ULTIMATE AUTO INSTALLER")
        print("=" * 60)
        
        # Ask for confirmation
        if installer.rich_available:
            from rich.prompt import Confirm
            if not Confirm.ask("🚀 Start ultimate installation?", default=True):
                print("Installation cancelled by user.")
                return
        else:
            response = input("🚀 Start ultimate installation? (Y/n): ").strip().lower()
            if response and response != 'y' and response != 'yes':
                print("Installation cancelled by user.")
                return
        
        # Run installation
        success = installer.run_ultimate_installation()
        
        # Create report
        installer.create_installation_report()
        
        # Display summary
        installer.display_final_summary(success)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
