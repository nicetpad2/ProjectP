#!/usr/bin/env python3
"""
🛠️ CORE SAFE UTILITIES - CENTRALIZED UTILITY FUNCTIONS
Centralized utility functions to eliminate code duplication across the project

🎯 PURPOSE:
✅ Centralized safe_print function for all modules
✅ Cross-platform compatibility (Windows/Linux/macOS)
✅ BrokenPipeError protection (Colab/Jupyter safe)
✅ Unicode handling for terminal output
✅ Error resilient output functions

⭐ ELIMINATES DUPLICATION:
- Replaces 80+ duplicate safe_print implementations
- Single source of truth for output utilities
- Consistent error handling across project
"""

import sys
import os
from typing import Any, Optional


def safe_print(*args, **kwargs) -> None:
    """
    Safe print function with comprehensive error handling
    
    Features:
    - BrokenPipeError protection (Colab/Jupyter safe)
    - UnicodeEncodeError handling (Windows safe)
    - Cross-platform compatibility
    - Graceful fallback to stderr
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    try:
        # Try normal print first
        print(*args, **kwargs)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        # Handle broken pipe (common in Jupyter/Colab)
        try:
            # Fallback to stderr
            message = " ".join(map(str, args))
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            # Ultimate fallback - silent failure
            pass
    except UnicodeEncodeError:
        # Handle unicode errors (Windows compatibility)
        try:
            # Convert to ASCII with error replacement
            message = " ".join(str(arg).encode('ascii', 'replace').decode('ascii') for arg in args)
            print(message, **kwargs)
            sys.stdout.flush()
        except Exception:
            # Fallback to basic ASCII
            try:
                message = " ".join(str(arg) for arg in args if ord(max(str(arg), default=' ')) < 128)
                print(message, **kwargs)
                sys.stdout.flush()
            except Exception:
                pass
    except Exception:
        # Handle any other errors
        try:
            # Basic fallback
            message = " ".join(str(arg) for arg in args)
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            # Silent failure as last resort
            pass


def safe_input(prompt: str = "", fallback: str = "") -> str:
    """
    Safe input function with error handling
    
    Args:
        prompt: Input prompt message
        fallback: Fallback value if input fails
        
    Returns:
        User input or fallback value
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return fallback
    except Exception:
        return fallback


def safe_write_file(filepath: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Safe file writing with error handling
    
    Args:
        filepath: Path to file
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except Exception:
        return False


def safe_read_file(filepath: str, encoding: str = 'utf-8', fallback: str = "") -> str:
    """
    Safe file reading with error handling
    
    Args:
        filepath: Path to file
        encoding: File encoding
        fallback: Fallback content if read fails
        
    Returns:
        File content or fallback
    """
    try:
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    except Exception:
        return fallback


def safe_format_number(value: Any, decimal_places: int = 2, fallback: str = "N/A") -> str:
    """
    Safe number formatting with error handling
    
    Args:
        value: Value to format
        decimal_places: Number of decimal places
        fallback: Fallback string if formatting fails
        
    Returns:
        Formatted number string or fallback
    """
    try:
        if isinstance(value, (int, float)):
            return f"{value:.{decimal_places}f}"
        else:
            return str(float(value))
    except (ValueError, TypeError):
        return fallback


def safe_percentage(value: Any, total: Any, decimal_places: int = 1, fallback: str = "N/A") -> str:
    """
    Safe percentage calculation with error handling
    
    Args:
        value: Numerator value
        total: Denominator value
        decimal_places: Number of decimal places
        fallback: Fallback string if calculation fails
        
    Returns:
        Percentage string or fallback
    """
    try:
        if total and float(total) != 0:
            percentage = (float(value) / float(total)) * 100
            return f"{percentage:.{decimal_places}f}%"
        else:
            return "0.0%"
    except (ValueError, TypeError, ZeroDivisionError):
        return fallback


# Backwards compatibility aliases
print_safe = safe_print  # Alternative name
input_safe = safe_input  # Alternative name

# Export main functions
__all__ = [
    'safe_print',
    'safe_input', 
    'safe_write_file',
    'safe_read_file',
    'safe_format_number',
    'safe_percentage',
    'print_safe',
    'input_safe'
]