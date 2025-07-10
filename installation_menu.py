#!/usr/bin/env python3
"""
🎯 NICEGOLD ProjectP - Installation Menu
Interactive installation management system
"""

import os
import sys
import subprocess
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def install_rich_if_missing():
    """Install rich library if missing"""
    try:
        import rich
        from rich.console import Console
        return Console()
    except ImportError:
        print("📦 Installing 'rich' library...")
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
    from rich.prompt import Prompt, Confirm
    from rich.text import Text

def show_menu():
    """Display the installation menu"""
    if console:
        console.print(Panel.fit(
            "[bold cyan]🎯 NICEGOLD ProjectP Installation Menu[/bold cyan]",
            subtitle="[yellow]Choose your installation option[/yellow]"
        ))
        
        menu_table = Table(show_header=True, header_style="bold magenta")
        menu_table.add_column("Option", style="cyan", width=8)
        menu_table.add_column("Description", style="white")
        menu_table.add_column("Recommended", style="dim")
        
        menu_table.add_row("1", "🔍 Check Installation Status", "Always run first")
        menu_table.add_row("2", "📦 Install Basic Dependencies", "Quick install")
        menu_table.add_row("3", "🏢 Enterprise Installation", "Full feature install")
        menu_table.add_row("4", "🔧 Install Missing Packages Only", "Fix partial install")
        menu_table.add_row("5", "🌟 Complete System Setup", "Full setup + verification")
        menu_table.add_row("6", "❌ Exit", "")
        
        console.print(menu_table)
    else:
        print("🎯 NICEGOLD ProjectP Installation Menu")
        print("=" * 50)
        print("1. 🔍 Check Installation Status")
        print("2. 📦 Install Basic Dependencies")
        print("3. 🏢 Enterprise Installation")
        print("4. 🔧 Install Missing Packages Only")
        print("5. 🌟 Complete System Setup")
        print("6. ❌ Exit")
        print("=" * 50)

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        if console:
            console.print(f"[red]❌ Script not found: {script_name}[/red]")
        else:
            print(f"❌ Script not found: {script_name}")
        return False
    
    if console:
        console.print(f"[cyan]🚀 {description}...[/cyan]")
    else:
        print(f"🚀 {description}...")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            if console:
                console.print(f"[green]✅ {description} completed successfully![/green]")
            else:
                print(f"✅ {description} completed successfully!")
            return True
        else:
            if console:
                console.print(f"[yellow]⚠️ {description} completed with warnings.[/yellow]")
            else:
                print(f"⚠️ {description} completed with warnings.")
            return False
    except Exception as e:
        if console:
            console.print(f"[red]❌ Error running {description}: {e}[/red]")
        else:
            print(f"❌ Error running {description}: {e}")
        return False

def install_missing_packages():
    """Install only missing packages"""
    if console:
        console.print("[cyan]🔧 Installing missing packages only...[/cyan]")
    
    # First check what's missing
    if not run_script('check_installation.py', 'Checking current installation'):
        return False
    
    # Then install missing packages
    return run_script('install_dependencies.py', 'Installing missing dependencies')

def complete_system_setup():
    """Complete system setup with verification"""
    if console:
        console.print("[cyan]🌟 Starting complete system setup...[/cyan]")
    
    success = True
    
    # Step 1: Enterprise installation
    if console:
        console.print("[cyan]Step 1: Enterprise Installation[/cyan]")
    success &= run_script('install_enterprise.py', 'Enterprise installation')
    
    # Step 2: Install dependencies
    if console:
        console.print("[cyan]Step 2: Installing Dependencies[/cyan]")
    success &= run_script('install_dependencies.py', 'Installing dependencies')
    
    # Step 3: Final verification
    if console:
        console.print("[cyan]Step 3: Final Verification[/cyan]")
    success &= run_script('check_installation.py', 'Final installation check')
    
    if success:
        if console:
            console.print("[bold green]🎉 Complete system setup finished successfully![/bold green]")
            console.print("[cyan]ℹ️ You can now run: python ProjectP.py[/cyan]")
        else:
            print("🎉 Complete system setup finished successfully!")
            print("ℹ️ You can now run: python ProjectP.py")
    else:
        if console:
            console.print("[yellow]⚠️ System setup completed with some issues.[/yellow]")
        else:
            print("⚠️ System setup completed with some issues.")
    
    return success

def main():
    """Main menu loop"""
    while True:
        print("\n")
        show_menu()
        
        if console:
            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5", "6"], default="1")
        else:
            choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            run_script('check_installation.py', 'Checking installation status')
        
        elif choice == "2":
            run_script('install_dependencies.py', 'Installing basic dependencies')
        
        elif choice == "3":
            run_script('install_enterprise.py', 'Enterprise installation')
        
        elif choice == "4":
            install_missing_packages()
        
        elif choice == "5":
            complete_system_setup()
        
        elif choice == "6":
            if console:
                console.print("[cyan]👋 Thank you for using NICEGOLD ProjectP![/cyan]")
            else:
                print("👋 Thank you for using NICEGOLD ProjectP!")
            break
        
        else:
            if console:
                console.print("[red]❌ Invalid choice. Please select 1-6.[/red]")
            else:
                print("❌ Invalid choice. Please select 1-6.")
        
        # Ask if user wants to continue
        if console:
            if not Confirm.ask("\nDo you want to continue?", default=True):
                break
        else:
            continue_choice = input("\nDo you want to continue? (y/n): ").strip().lower()
            if continue_choice in ['n', 'no']:
                break

if __name__ == "__main__":
    main()
