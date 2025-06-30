#!/usr/bin/env python3
"""
📁 NICEGOLD OUTPUT MANAGER
ระบบจัดการ Output ให้เป็นระเบียบเรียบร้อย
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import logging

class NicegoldOutputManager:
    """ตัวจัดการ Output ให้เป็นระเบียบ"""
    
    def __init__(self, base_path: str = "outputs"):
        self.base_path = base_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """สร้างโครงสร้างโฟลเดอร์ output"""
        directories = [
            self.base_path,
            f"{self.base_path}/models",
            f"{self.base_path}/results", 
            f"{self.base_path}/reports",
            f"{self.base_path}/charts",
            f"{self.base_path}/data",
            f"{self.base_path}/logs",
            f"{self.base_path}/analysis",
            f"{self.base_path}/sessions",
            f"{self.base_path}/sessions/{self.session_id}"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        self.logger.info(f"📁 Output directories created for session: {self.session_id}")
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> str:
        """บันทึกโมเดล ML"""
        try:
            # Model file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = f"{self.base_path}/models/{model_name}_{timestamp}.joblib"
            
            # Import joblib for saving
            import joblib
            joblib.dump(model, model_file)
            
            # Save metadata
            if metadata:
                metadata_file = f"{self.base_path}/models/{model_name}_{timestamp}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Model saved: {os.path.basename(model_file)}")
            return model_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {str(e)}")
            return ""
    
    def save_results(self, results: Dict, result_name: str) -> str:
        """บันทึกผลลัพธ์"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{self.base_path}/results/{result_name}_{timestamp}.json"
            
            # Make results JSON serializable
            clean_results = self._make_json_serializable(results)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Results saved: {os.path.basename(results_file)}")
            return results_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}")
            return ""
    
    def save_data(self, data: pd.DataFrame, data_name: str, format: str = "csv") -> str:
        """บันทึกข้อมูล"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "csv":
                data_file = f"{self.base_path}/data/{data_name}_{timestamp}.csv"
                data.to_csv(data_file, index=False)
            elif format.lower() == "json":
                data_file = f"{self.base_path}/data/{data_name}_{timestamp}.json"
                data.to_json(data_file, orient='records', indent=2)
            elif format.lower() == "parquet":
                data_file = f"{self.base_path}/data/{data_name}_{timestamp}.parquet"
                data.to_parquet(data_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"🗂️ Data saved: {os.path.basename(data_file)} ({len(data):,} rows)")
            return data_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save data: {str(e)}")
            return ""
    
    def save_chart(self, figure, chart_name: str, format: str = "png") -> str:
        """บันทึกกราฟ/แผนภูมิ"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"{self.base_path}/charts/{chart_name}_{timestamp}.{format}"
            
            # Save chart based on type
            if hasattr(figure, 'savefig'):  # matplotlib figure
                figure.savefig(chart_file, dpi=300, bbox_inches='tight')
            elif hasattr(figure, 'write_image'):  # plotly figure
                figure.write_image(chart_file)
            else:
                # Try to save as matplotlib
                figure.savefig(chart_file, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"📈 Chart saved: {os.path.basename(chart_file)}")
            return chart_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save chart: {str(e)}")
            return ""
    
    def generate_report(self, title: str, content: Dict, template: str = "basic") -> str:
        """สร้างรายงาน HTML"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{self.base_path}/reports/{title}_{timestamp}.html"
            
            html_content = self._generate_html_report(title, content, template)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"📋 Report generated: {os.path.basename(report_file)}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {str(e)}")
            return ""
    
    def save_session_summary(self, summary: Dict) -> str:
        """บันทึกสรุปของ session"""
        try:
            summary_file = f"{self.base_path}/sessions/{self.session_id}/session_summary.json"
            
            # Add session metadata
            summary_with_meta = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "summary": summary
            }
            
            clean_summary = self._make_json_serializable(summary_with_meta)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(clean_summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📝 Session summary saved: {self.session_id}")
            return summary_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save session summary: {str(e)}")
            return ""
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """แปลง object ให้เป็น JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _generate_html_report(self, title: str, content: Dict, template: str) -> str:
        """สร้าง HTML report"""
        html = f"""
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid #4ecdc4;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px 20px;
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
            min-width: 120px;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 {title}</h1>
            <p>NICEGOLD Enterprise Elliott Wave Report</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        <div class="content">
"""
        
        # Add content sections
        for section_name, section_data in content.items():
            html += f"""
            <div class="section">
                <h2>{section_name}</h2>
"""
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    html += f"""
                <div class="metric">
                    <div>{key}</div>
                    <div style="font-size: 1.2em; margin-top: 5px;">{value}</div>
                </div>
"""
            else:
                html += f"<p>{section_data}</p>"
            
            html += "</div>"
        
        html += f"""
        </div>
        <div class="footer">
            <p>🏆 NICEGOLD Enterprise System - Session ID: {self.session_id}</p>
            <p>© 2024 NICEGOLD ProjectP - Elliott Wave Analysis</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def get_session_path(self) -> str:
        """ได้ path ของ session ปัจจุบัน"""
        return f"{self.base_path}/sessions/{self.session_id}"
    
    def list_outputs(self) -> Dict[str, list]:
        """แสดงรายการ output ทั้งหมด"""
        try:
            outputs = {}
            
            for subdir in ['models', 'results', 'reports', 'charts', 'data']:
                full_path = f"{self.base_path}/{subdir}"
                if os.path.exists(full_path):
                    outputs[subdir] = [f for f in os.listdir(full_path) if not f.startswith('.')]
                else:
                    outputs[subdir] = []
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list outputs: {str(e)}")
            return {}
