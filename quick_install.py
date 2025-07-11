#!/usr/bin/env python3
"""
🚀 NICEGOLD ProjectP - Quick Install System
Fast and efficient dependency installation for NICEGOLD Enterprise ProjectP
This script provides the quickest way to install all required dependencies
"""

import subprocess
import sys
import os
import platform
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Set UTF-8 encoding for cross-platform compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

def check_python_version() -> bool:
    """Check if Python version is supported"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (supported)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def install_rich_for_ui():
    """Install rich library for better UI"""
    try:
        import rich
        from rich.console import Console
        return Console()
    except ImportError:
        print("📦 Installing 'rich' library for better UI...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rich'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from rich.console import Console
            return Console()
        except Exception as e:
            print(f"⚠️ Could not install 'rich': {e}")
            return None

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    return {
        'platform': platform.system(),
        'arch': platform.machine(),
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pip': 'available' if check_pip() else 'not available'
    }

def check_pip() -> bool:
    """Check if pip is available"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    try:
        print("📦 Upgrading pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                            stdout=subprocess.DEVNULL)
        print("✅ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Could not upgrade pip")
        return False

def get_requirements_file() -> str | None:
    """Find the best requirements file to use"""
    priority_files = [
        'requirements_complete.txt',  # Most comprehensive
        'requirements.txt'            # Standard
    ]
    
    for req_file in priority_files:
        if Path(req_file).exists():
            return req_file
    
    return None

def parse_requirements(filepath: str) -> List[str]:
    """Parse requirements file and return clean package list"""
    packages = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        packages.append(line)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return []
    
    return packages

def install_package(package: str, timeout: int = 300) -> bool:
    """Install a single package with timeout"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout installing {package}")
        return False
    except Exception:
        return False

def install_packages_batch(packages: List[str], batch_size: int = 5) -> Tuple[List[str], List[str]]:
    """Install packages in batches for better performance"""
    successful = []
    failed = []
    
    for i in range(0, len(packages), batch_size):
        batch = packages[i:i + batch_size]
        print(f"📦 Installing batch {i//batch_size + 1}: {', '.join(batch)}")
        
        for package in batch:
            if install_package(package):
                successful.append(package)
                print(f"  ✅ {package}")
            else:
                failed.append(package)
                print(f"  ❌ {package}")
    
    return successful, failed

def create_installation_report(successful: List[str], failed: List[str], requirements_file: str):
    """Create installation report"""
    total = len(successful) + len(failed)
    success_rate = (len(successful) / total * 100) if total > 0 else 0
    
    report = f"""
🏢 NICEGOLD ProjectP - Quick Installation Report
{'=' * 60}
Requirements File: {requirements_file}
Total Packages: {total}
Successful: {len(successful)} ({success_rate:.1f}%)
Failed: {len(failed)} ({100 - success_rate:.1f}%)
{'=' * 60}
"""
    
    if failed:
        report += "\n❌ Failed Packages:\n"
        for pkg in failed:
            report += f"  - {pkg}\n"
    
    report += f"\n🕒 Installation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Save report to file
    try:
        with open('installation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("📄 Installation report saved to: installation_report.txt")
    except Exception:
        pass
    
    return report

def main():
    """Main quick installation function"""
    
    print("🚀 NICEGOLD ProjectP - Quick Install System")
    print("=" * 60)
    print("⚡ Fast dependency installation for enterprise AI trading system")
    print()
    
    # System checks
    print("🔍 System Checks:")
    if not check_python_version():
        print("❌ Python version not supported. Please upgrade to Python 3.8+")
        return False
    
    if not check_pip():
        print("❌ pip not available. Please install pip first.")
        return False
    
    # Show system info
    system_info = get_system_info()
    print(f"💻 Platform: {system_info['platform']} {system_info['arch']}")
    print(f"🐍 Python: {system_info['python']}")
    print()
    
    # Upgrade pip
    upgrade_pip()
    print()
    
    # Install rich for better UI
    console = install_rich_for_ui()
    
    # Find requirements file
    requirements_file = get_requirements_file()
    if not requirements_file:
        print("❌ No requirements file found!")
        print("📄 Please ensure either requirements.txt or requirements_complete.txt exists")
        return False
    
    print(f"📋 Using requirements file: {requirements_file}")
    
    # Parse requirements
    packages = parse_requirements(requirements_file)
    if not packages:
        print("❌ No packages found in requirements file!")
        return False
    
    print(f"📦 Found {len(packages)} packages to install")
    print()
    
    # Install packages
    print("🚀 Starting installation...")
    start_time = time.time()
    
    if console:
        # Use rich for better progress display
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Installing packages...", total=len(packages))
            
            successful = []
            failed = []
            
            for package in packages:
                progress.update(task, description=f"Installing {package}...")
                if install_package(package):
                    successful.append(package)
                else:
                    failed.append(package)
                progress.advance(task)
    else:
        # Simple progress without rich
        successful, failed = install_packages_batch(packages)
    
    # Calculate time
    elapsed = time.time() - start_time
    
    # Create and display report
    print()
    report = create_installation_report(successful, failed, requirements_file)
    print(report)
    
    print(f"⏱️ Installation completed in {elapsed:.1f} seconds")
    
    # Final recommendations
    if len(failed) == 0:
        print("🎉 All packages installed successfully!")
        print("✅ Your NICEGOLD ProjectP system is ready!")
        print("🚀 You can now run: python ProjectP.py")
    else:
        print(f"⚠️ Installation completed with {len(failed)} failed packages")
        print("🔧 You can try:")
        print("   1. Run: python install_dependencies.py (for individual package retry)")
        print("   2. Run: python installation_menu.py (for interactive installation)")
        print("   3. Install failed packages manually")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Quick installation completed successfully!")
        exit_code = 0
    else:
        print("⚠️ Quick installation completed with issues.")
        exit_code = 1
    
    print("📚 For more installation options, run: python installation_menu.py")
    sys.exit(exit_code)
