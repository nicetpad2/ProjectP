#!/usr/bin/env python3
"""
📁 NICEGOLD OUTPUT MANAGER
ระบบจัดการ Output ให้เป็นระเบียบเรียบร้อย - ใช้ ProjectPaths
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import logging
from .project_paths import get_project_paths


class NicegoldOutputManager:
    """ตัวจัดการ Output ให้เป็นระเบียบ - ใช้ ProjectPaths"""
    
    def __init__(self, use_project_paths: bool = True):
        self.logger = get_unified_logger()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if use_project_paths:
            # ใช้ ProjectPaths สำหรับจัดการ path
            self.paths = get_project_paths()
            self.use_paths = True
        else:
            # Fallback to legacy mode
            self.paths = None
            self.use_paths = False
        
        # Create directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """สร้างโครงสร้างโฟลเดอร์ output"""
        if self.use_paths and self.paths:
            # ใช้ ProjectPaths - directories จะถูกสร้างอัตโนมัติ
            session_dir = self.paths.outputs / "sessions" / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            log_msg = f"📁 Output directories ready for session: {self.session_id}"
            self.logger.info(log_msg)
        else:
            self.logger.error("❌ ProjectPaths not available")
    
    def save_model(self, model: Any, model_name: str, 
                   metadata: Optional[Dict] = None) -> str:
        """บันทึกโมเดล ML"""
        try:
            if self.use_paths and self.paths:
                # ใช้ ProjectPaths
                model_file = self.paths.get_model_file_path(model_name)
            else:
                self.logger.error("❌ ProjectPaths not available")
                return ""
            
            # Import joblib for saving
            import joblib
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

            joblib.dump(model, model_file)
            
            # Save metadata
            if metadata:
                metadata_file = str(model_file).replace('.joblib', 
                                                      '_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Model saved: {model_file.name}")
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {str(e)}")
            return ""
    
    def save_results(self, results: Dict, result_name: str) -> str:
        """บันทึกผลลัพธ์"""
        try:
            if self.use_paths and self.paths:
                results_file = self.paths.get_results_file_path(result_name)
            else:
                self.logger.error("❌ ProjectPaths not available")
                return ""
            
            # Make results JSON serializable
            clean_results = self._make_json_serializable(results)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Results saved: {results_file.name}")
            return str(results_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}")
            return ""
    
    def save_data(self, data: pd.DataFrame, data_name: str, 
                  format: str = "csv") -> str:
        """บันทึกข้อมูล"""
        try:
            if not self.use_paths or not self.paths:
                self.logger.error("❌ ProjectPaths not available")
                return ""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "csv":
                data_file = (self.paths.outputs / "data" / 
                           f"{data_name}_{timestamp}.csv")
                data.to_csv(data_file, index=False)
            elif format.lower() == "json":
                data_file = (self.paths.outputs / "data" / 
                           f"{data_name}_{timestamp}.json")
                data.to_json(data_file, orient='records', indent=2)
            elif format.lower() == "parquet":
                data_file = (self.paths.outputs / "data" / 
                           f"{data_name}_{timestamp}.parquet")
                data.to_parquet(data_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            row_count = len(data)
            log_msg = f"🗂️ Data saved: {data_file.name} ({row_count:,} rows)"
            self.logger.info(log_msg)
            return str(data_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save data: {str(e)}")
            return ""
    
    def save_chart(self, figure, chart_name: str, format: str = "png") -> str:
        """บันทึกกราฟ/แผนภูมิ"""
        try:
            if not self.use_paths or not self.paths:
                self.logger.error("❌ ProjectPaths not available")
                return ""
            
            chart_file = self.paths.get_chart_file_path(chart_name, format)
            
            # Save based on figure type
            if hasattr(figure, 'savefig'):
                # Matplotlib figure
                figure.savefig(chart_file, dpi=300, bbox_inches='tight')
            elif hasattr(figure, 'write_image'):
                # Plotly figure
                figure.write_image(str(chart_file))
            else:
                raise ValueError("Unsupported figure type")
            
            self.logger.info(f"📈 Chart saved: {chart_file.name}")
            return str(chart_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save chart: {str(e)}")
            return ""
    
    def save_report(self, content: str, report_name: str, 
                   format: str = "txt") -> str:
        """บันทึกรายงาน"""
        try:
            if not self.use_paths or not self.paths:
                self.logger.error("❌ ProjectPaths not available")
                return ""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (self.paths.reports / 
                          f"{report_name}_{timestamp}.{format}")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"📄 Report saved: {report_file.name}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save report: {str(e)}")
            return ""
    
    def _make_json_serializable(self, obj):
        """แปลงให้เป็น JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif str(type(obj)).startswith('<class \'numpy.'):
            return str(obj)
        else:
            return obj
    
    def get_session_summary(self) -> Dict[str, Any]:
        """ดึงข้อมูลสรุปของ session"""
        if not self.use_paths or not self.paths:
            return {"error": "ProjectPaths not available"}
        
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "paths": {
                "project_root": str(self.paths.project_root),
                "outputs": str(self.paths.outputs),
                "models": str(self.paths.models),
                "results": str(self.paths.results),
                "charts": str(self.paths.charts),
                "reports": str(self.paths.reports)
            },
            "data_files": self.paths.list_data_files() if self.paths else []
        }
        
        return summary
    
    def cleanup_temp_files(self):
        """ลบไฟล์ temp ที่เก่า"""
        try:
            if not self.use_paths or not self.paths:
                return
            
            temp_files = list(self.paths.temp.glob("*"))
            for temp_file in temp_files:
                if temp_file.is_file():
                    temp_file.unlink()
            
            self.logger.info(f"🧹 Cleaned up {len(temp_files)} temp files")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup temp files: {str(e)}")


# Convenience function
def create_output_manager() -> NicegoldOutputManager:
    """สร้าง OutputManager ใหม่"""
    return NicegoldOutputManager(use_project_paths=True)


if __name__ == "__main__":
    # Test the output manager
    manager = create_output_manager()
    
    print("🏢 NICEGOLD Enterprise Output Manager")
    print("=" * 50)
    
    summary = manager.get_session_summary()
    print(f"Session ID: {summary['session_id']}")
    print(f"Timestamp: {summary['timestamp']}")
    print()
    
    print("📁 Paths:")
    for key, value in summary['paths'].items():
        print(f"  {key:12}: {value}")
    
    print()
    print("📊 Data Files:")
    for file in summary['data_files']:
        print(f"  - {file}")
