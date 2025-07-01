#!/usr/bin/env python3
"""
📁 NICEGOLD ENTERPRISE PROJECT PATHS
ระบบจัดการ path ทุกสภาพแวดล้อม - Windows, Linux, MacOS
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import logging


class ProjectPaths:
    """Enterprise Project Path Manager"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize Project Paths
        
        Args:
            base_path: Base project directory (auto-detect if None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect project root
        if base_path is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(base_path).resolve()
        
        # Validate project structure
        self._validate_project_structure()
        
        # Initialize all paths
        self._initialize_paths()
        
        self.logger.info(f"📁 Project paths initialized: {self.project_root}")
    
    def _find_project_root(self) -> Path:
        """อัตโนมัติหา project root directory"""
        # Start from current file location
        current_file = Path(__file__).resolve()
        
        # Look for project markers
        project_markers = [
            'ProjectP.py',
            'requirements.txt',
            'config',
            'core',
            'elliott_wave_modules'
        ]
        
        # Search up the directory tree
        search_parents = [current_file.parent.parent] + list(
            current_file.parents
        )
        for parent in search_parents:
            markers_exist = all(
                parent.joinpath(marker).exists()
                for marker in project_markers[:3]
            )
            if markers_exist:
                return parent
        
        # Fallback to parent of core directory
        return current_file.parent.parent
    
    def _validate_project_structure(self):
        """ตรวจสอบโครงสร้างโปรเจค"""
        required_dirs = [
            'core',
            'config',
            'elliott_wave_modules',
            'menu_modules',
            'datacsv'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.logger.warning(f"⚠️  Directory not found: {dir_path}")
    
    def _initialize_paths(self):
        """สร้าง path สำหรับโฟลเดอร์ต่างๆ"""
        # Core directories (existing)
        self.core = self.project_root / 'core'
        self.config = self.project_root / 'config'
        self.elliott_wave_modules = self.project_root / 'elliott_wave_modules'
        self.menu_modules = self.project_root / 'menu_modules'
        
        # Data directories
        self.datacsv = self.project_root / 'datacsv'
        self.models = self.project_root / 'models'
        self.results = self.project_root / 'results'
        self.logs = self.project_root / 'logs'
        self.temp = self.project_root / 'temp'
        
        # Output directories (will be created as needed)
        self.outputs = self.project_root / 'outputs'
        self.reports = self.outputs / 'reports'
        self.charts = self.outputs / 'charts'
        self.analysis = self.outputs / 'analysis'
        
        # Cache directories
        self.cache = self.project_root / '.cache'
        self.venv = self.project_root / '.venv'
        
        # Config files
        self.enterprise_config = self.config / 'enterprise_config.yaml'
        self.requirements_file = self.project_root / 'requirements.txt'
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """สร้างโฟลเดอร์ที่จำเป็น"""
        directories_to_create = [
            self.models,
            self.results,
            self.logs,
            self.temp,
            self.outputs,
            self.reports,
            self.charts,
            self.analysis,
            self.cache
        ]
        
        for directory in directories_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"📁 Created directory: {directory}")
            except Exception as e:
                error_msg = f"❌ Failed to create directory {directory}: {e}"
                self.logger.error(error_msg)
    
    def get_path(self, path_key: str) -> Path:
        """
        ดึง path ตามชื่อ key
        
        Args:
            path_key: ชื่อ path (เช่น 'models', 'results', 'datacsv')
            
        Returns:
            Path object
        """
        if hasattr(self, path_key):
            return getattr(self, path_key)
        else:
            raise ValueError(f"Unknown path key: {path_key}")
    
    def get_data_file_path(self, filename: str) -> Path:
        """ดึง path ของไฟล์ data"""
        return self.datacsv / filename
    
    def get_model_file_path(self, model_name: str,
                            timestamp: Optional[str] = None) -> Path:
        """ดึง path ของไฟล์ model"""
        if timestamp is None:
            timestamp = self._get_timestamp()
        return self.models / f"{model_name}_{timestamp}.joblib"
    
    def get_results_file_path(self, result_name: str,
                              timestamp: Optional[str] = None) -> Path:
        """ดึง path ของไฟล์ results"""
        if timestamp is None:
            timestamp = self._get_timestamp()
        return self.results / f"{result_name}_{timestamp}.json"
    
    def get_log_file_path(self, log_name: str = "nicegold_enterprise") -> Path:
        """ดึง path ของไฟล์ log"""
        timestamp = self._get_date_timestamp()
        return self.logs / f"{log_name}_{timestamp}.log"
    
    def get_temp_file_path(self, filename: str) -> Path:
        """ดึง path ของไฟล์ temp"""
        return self.temp / filename
    
    def get_chart_file_path(self, chart_name: str, format: str = "png",
                            timestamp: Optional[str] = None) -> Path:
        """ดึง path ของไฟล์ chart"""
        if timestamp is None:
            timestamp = self._get_timestamp()
        return self.charts / f"{chart_name}_{timestamp}.{format}"
    
    def _get_timestamp(self) -> str:
        """สร้าง timestamp สำหรับไฟล์"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_date_timestamp(self) -> str:
        """สร้าง date timestamp สำหรับไฟล์"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")
    
    def ensure_directory_exists(self, directory_path: Path) -> bool:
        """ตรวจสอบและสร้างโฟลเดอร์ถ้าไม่มี"""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            error_msg = f"❌ Failed to create directory {directory_path}: {e}"
            self.logger.error(error_msg)
            return False
    
    def get_relative_path(self, absolute_path: Path) -> Path:
        """แปลง absolute path เป็น relative path จาก project root"""
        try:
            return absolute_path.relative_to(self.project_root)
        except ValueError:
            return absolute_path
    
    def is_valid_data_file(self, filename: str) -> bool:
        """ตรวจสอบว่าไฟล์ data มีอยู่จริง"""
        file_path = self.get_data_file_path(filename)
        return file_path.exists() and file_path.is_file()
    
    def list_data_files(self, extension: str = ".csv") -> list:
        """แสดงรายการไฟล์ data"""
        if not self.datacsv.exists():
            return []
        
        return [f.name for f in self.datacsv.glob(f"*{extension}")]
    
    def get_system_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลระบบ"""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "python_version": sys.version,
            "project_root": str(self.project_root),
            "working_directory": os.getcwd(),
            "path_separator": os.sep,
            "user_home": str(Path.home())
        }
    
    def to_dict(self) -> Dict[str, str]:
        """แปลงเป็น dictionary สำหรับใช้ใน configuration"""
        return {
            "project_root": str(self.project_root),
            "core": str(self.core),
            "config": str(self.config),
            "datacsv": str(self.datacsv),
            "models": str(self.models),
            "results": str(self.results),
            "logs": str(self.logs),
            "temp": str(self.temp),
            "outputs": str(self.outputs),
            "reports": str(self.reports),
            "charts": str(self.charts),
            "analysis": str(self.analysis),
            "cache": str(self.cache)
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"ProjectPaths(root={self.project_root})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"ProjectPaths(project_root='{self.project_root}')"


# Global instance สำหรับใช้งานทั่วไป
project_paths = ProjectPaths()


def get_project_paths() -> ProjectPaths:
    """ดึง global ProjectPaths instance"""
    return project_paths


def get_data_file_path(filename: str) -> Path:
    """Convenience function สำหรับดึง data file path"""
    return project_paths.get_data_file_path(filename)


def get_model_file_path(model_name: str,
                        timestamp: Optional[str] = None) -> Path:
    """Convenience function สำหรับดึง model file path"""
    return project_paths.get_model_file_path(model_name, timestamp)


def get_results_file_path(result_name: str,
                          timestamp: Optional[str] = None) -> Path:
    """Convenience function สำหรับดึง results file path"""
    return project_paths.get_results_file_path(result_name, timestamp)


def get_log_file_path(log_name: str = "nicegold_enterprise") -> Path:
    """Convenience function สำหรับดึง log file path"""
    return project_paths.get_log_file_path(log_name)


if __name__ == "__main__":
    # Test the path system
    paths = ProjectPaths()
    
    print("🏢 NICEGOLD Enterprise Project Paths")
    print("=" * 50)
    print(f"Project Root: {paths.project_root}")
    print(f"Platform: {platform.system()}")
    print()
    
    print("📁 Directory Structure:")
    for key, value in paths.to_dict().items():
        exists = "✅" if Path(value).exists() else "❌"
        print(f"  {key:15}: {value} {exists}")
    
    print()
    print("📊 Data Files:")
    data_files = paths.list_data_files()
    if data_files:
        for file in data_files:
            print(f"  - {file}")
    else:
        print("  No data files found")
    
    print()
    print("ℹ️  System Info:")
    for key, value in paths.get_system_info().items():
        print(f"  {key}: {value}")
