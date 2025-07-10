@echo off
rem This batch file ensures that the console's code page is set to UTF-8 (65001)
rem before running the Python dependency installer. This is the most reliable way
rem to prevent UnicodeErrors when using rich-text libraries like 'rich' on Windows.

chcp 65001 > nul

echo Running NICEGOLD Enterprise ProjectP Dependency Installer...
python install_dependencies.py

pause 