#!/usr/bin/env python3
"""
🔧 NICEGOLD PROJECT PATH UTILITIES
Utility functions for consistent path handling across all modules
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, List


def ensure_project_in_path(project_name: str = "ProjectP") -> Path:
    """
    Ensure project root is in sys.path and return project root path
    
    Args:
        project_name: Name of the project directory to find
        
    Returns:
        Path to project root
    """
    # Get current file's directory
    current_file = Path(__file__).resolve()
    
    # Search for project root
    for parent in [current_file.parent.parent] + list(current_file.parents):
        if parent.name == project_name or (parent / "ProjectP.py").exists():
            project_root = parent
            break
    else:
        # Fallback to parent of core directory
        project_root = current_file.parent.parent
    
    # Add to sys.path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_safe_project_paths():
    """
    Safely import and return project paths
    
    Returns:
        ProjectPaths instance or None if import fails
    """
    try:
        # Ensure project is in path
        ensure_project_in_path()
        
        # Import and return project paths
        from core.project_paths import get_project_paths
        return get_project_paths()
    except ImportError as e:
        print(f"Warning: Could not import project_paths: {e}")
        return None


def resolve_data_path(filename: str) -> Path:
    """
    Resolve path to data file with fallback options
    
    Args:
        filename: Name of the data file
        
    Returns:
        Path to data file
    """
    paths = get_safe_project_paths()
    
    if paths:
        return paths.get_data_file_path(filename)
    else:
        # Fallback to relative path
        project_root = ensure_project_in_path()
        return project_root / "datacsv" / filename


def resolve_model_path(model_name: str, timestamp: Optional[str] = None) -> Path:
    """
    Resolve path to model file with fallback options
    
    Args:
        model_name: Name of the model
        timestamp: Optional timestamp
        
    Returns:
        Path to model file
    """
    paths = get_safe_project_paths()
    
    if paths:
        return paths.get_model_file_path(model_name, timestamp)
    else:
        # Fallback to relative path
        project_root = ensure_project_in_path()
        if timestamp:
            filename = f"{model_name}_{timestamp}.joblib"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.joblib"
        return project_root / "models" / filename


def resolve_results_path(result_name: str, timestamp: Optional[str] = None) -> Path:
    """
    Resolve path to results file with fallback options
    
    Args:
        result_name: Name of the result
        timestamp: Optional timestamp
        
    Returns:
        Path to results file
    """
    paths = get_safe_project_paths()
    
    if paths:
        return paths.get_results_file_path(result_name, timestamp)
    else:
        # Fallback to relative path
        project_root = ensure_project_in_path()
        if timestamp:
            filename = f"{result_name}_{timestamp}.json"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result_name}_{timestamp}.json"
        return project_root / "results" / filename


def list_data_files(extension: str = ".csv") -> List[str]:
    """
    List data files with fallback options
    
    Args:
        extension: File extension to filter
        
    Returns:
        List of data file names
    """
    paths = get_safe_project_paths()
    
    if paths:
        return paths.list_data_files(extension)
    else:
        # Fallback to direct search
        project_root = ensure_project_in_path()
        datacsv_path = project_root / "datacsv"
        if datacsv_path.exists():
            return [f.name for f in datacsv_path.glob(f"*{extension}")]
        else:
            return []


def ensure_directory_exists(directory: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if necessary
    
    Args:
        directory: Directory path to create
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Warning: Could not create directory {directory}: {e}")
        return False


def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path to project root
    """
    return ensure_project_in_path()


if __name__ == "__main__":
    # Test the utilities
    print("🔧 Testing Project Path Utilities")
    print("=" * 50)
    
    # Test project root detection
    project_root = get_project_root()
    print(f"Project Root: {project_root}")
    
    # Test safe paths import
    paths = get_safe_project_paths()
    if paths:
        print(f"✅ ProjectPaths imported successfully")
        print(f"Data directory: {paths.datacsv}")
        print(f"Models directory: {paths.models}")
    else:
        print("⚠️  ProjectPaths import failed, using fallbacks")
    
    # Test data files listing
    data_files = list_data_files()
    print(f"Data files found: {len(data_files)}")
    for file in data_files[:3]:  # Show first 3
        print(f"  - {file}")
    
    print("\n✅ Path utilities test completed")
