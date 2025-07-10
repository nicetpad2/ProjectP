# install_dependencies.py
# 🚀 NICEGOLD Enterprise ProjectP - Unified Dependency Installer
# This script provides a robust, cross-platform, and user-friendly way to install
# all necessary dependencies from requirements.txt.
# It uses the 'rich' library to display beautiful progress and status updates.

import subprocess
import sys
import os
import time
from typing import List, Tuple

# --- Start of Fix for Unicode Errors ---
# Set the PYTHONIOENCODING environment variable to 'utf-8'.
# This is a robust way to ensure that subprocesses (like pip) use the correct
# encoding, which prevents UnicodeDecodeError on Windows systems.
os.environ['PYTHONIOENCODING'] = 'utf-8'
# --- End of Fix ---


def check_pip():
    """Checks if pip is available."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def install_rich_if_missing():
    """Installs the 'rich' library if it's not already installed, as it's needed for the UI."""
    try:
        import rich
        from rich.console import Console
        return Console()
    except ImportError:
        print(" 'rich' library not found. Installing it first for a better experience...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rich'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from rich.console import Console
            console = Console()
            console.print("[green]✓[/green] 'rich' installed successfully.")
            return console
        except Exception as e:
            print(f"FATAL: Could not install 'rich'. Please install it manually ('pip install rich') and rerun. Error: {e}")
            sys.exit(1)

console = install_rich_if_missing()

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

def get_dependencies_from_file(filepath: str) -> List[str]:
    """Reads the requirements.txt file, ignoring comments and empty lines."""
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error:[/bold red] '{filepath}' not found!")
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        dependencies = []
        for line in f:
            # Strip whitespace and then remove any inline comments
            line = line.strip()
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            
            # Add to list only if the line is not empty after stripping comments
            if line:
                dependencies.append(line)
    return dependencies

def install_package(package_name: str, progress: Progress, task_id) -> bool:
    """Installs a single package using pip and updates the progress bar."""
    progress.update(task_id, description=f"Installing {package_name}...")
    try:
        # Using subprocess.run for more control
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True, text=True, check=True
        )
        progress.update(task_id, advance=1)
        return True
    except subprocess.CalledProcessError as e:
        progress.stop()
        console.print(f"\n[bold red]Error installing {package_name}:[/bold red]")
        console.print(Text(e.stderr, style="red"))
        progress.start()
        progress.update(task_id, advance=1, description=f"[red]Failed: {package_name}[/red]")
        return False

def main():
    """Main function to run the dependency installation."""
    # Simplified the title to use only basic ASCII characters for max compatibility
    console.print(Panel.fit(
        "[bold cyan]NICEGOLD Enterprise ProjectP Dependency Installer[/bold cyan]",
        subtitle="[yellow]Ensuring a perfect project setup[/yellow]"
    ))

    if not check_pip():
        console.print("[bold red]FATAL: pip is not available. Please install pip to continue.[/bold red]")
        sys.exit(1)

    requirements_file = 'requirements.txt'
    dependencies = get_dependencies_from_file(requirements_file)
    
    if not dependencies:
        console.print(f"[yellow]Warning: No dependencies found in {requirements_file}.[/yellow]")
        sys.exit(0)
    
    console.print(f"\nFound [bold green]{len(dependencies)}[/bold green] packages to install from [cyan]{requirements_file}[/cyan].\n")

    successful_installs = []
    failed_installs = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False  # Keep the progress bar on screen after completion
    ) as progress:
        task = progress.add_task("[cyan]Installing dependencies...", total=len(dependencies))
        
        for dep in dependencies:
            time.sleep(0.1) # Small delay for better visual flow
            if install_package(dep, progress, task):
                successful_installs.append(dep)
            else:
                failed_installs.append(dep)

    # --- Summary ---
    console.print("\n" + "="*50)
    console.print("[bold cyan]Installation Summary[/bold cyan]")
    console.print("="*50 + "\n")

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Status", style="dim", width=12)
    summary_table.add_column("Count")

    summary_table.add_row("[green]Successful[/green]", str(len(successful_installs)))
    summary_table.add_row("[red]Failed[/red]", str(len(failed_installs)))
    console.print(summary_table)

    if failed_installs:
        console.print("\n[bold yellow]The following packages failed to install:[/bold yellow]")
        for pkg in failed_installs:
            console.print(f"- [red]{pkg}[/red]")
        console.print("\nPlease try installing them manually or check the error messages above.")
        console.print("[bold red]Project setup is incomplete.[/bold red]")
    else:
        console.print("\n[bold green]✅ All dependencies installed successfully![/bold green]")
        console.print("[bold cyan]Project setup is complete. You can now run the main application.[/bold cyan]")

if __name__ == "__main__":
    main()
