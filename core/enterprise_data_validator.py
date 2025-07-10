# -*- coding: utf-8 -*-
"""
/content/drive/MyDrive/ProjectP-1/core/enterprise_data_validator.py

## 🛡️ NICEGOLD Enterprise ProjectP - Enterprise Data Validator

**เวอร์ชัน:** 1.0
**สถานะ:** Production Ready
**อัปเดต:** 9 กรกฎาคม 2025
**ผู้พัฒนา:** NICEGOLD AI Agent (AgentP)
**ติดต่อ:** enterprise.nicegold.ai@nicegold.com

### 📝 **รายละเอียด**
โมดูลนี้เป็นระบบตรวจสอบและทำความสะอาดข้อมูลระดับ Enterprise สำหรับ NICEGOLD ProjectP 
ออกแบบมาเพื่อรับประกันความสมบูรณ์และความถูกต้องของข้อมูล (Data Integrity) ก่อนที่จะนำไปใช้ใน Pipeline การวิเคราะห์
และการฝึกโมเดล AI

### ✨ **ความสามารถหลัก (Key Features)**
- **Robust Type Enforcement:** บังคับใช้ประเภทข้อมูลที่ถูกต้องสำหรับคอลัมน์หลัก (OHLCV) อย่างเข้มงวด
- **Error Coercion:** แปลงค่าที่ผิดพลาดในคอลัมน์ตัวเลขให้เป็น `NaN` (Not a Number) เพื่อให้สามารถจัดการได้
- **Missing Value Handling:** ตรวจจับและลบแถวที่มีข้อมูลสำคัญขาดหายไป (Missing Values)
- **Detailed Logging:** บันทึกข้อมูลสถิติการตรวจสอบและการล้างข้อมูลอย่างละเอียดผ่าน Unified Enterprise Logger
- **Pre-computation Validation:** ตรวจสอบข้อมูลก่อนการคำนวณที่ซับซ้อน เช่น Noise Filtering เพื่อป้องกัน `TypeError`

### 🏗️ **สถาปัตยกรรม**
- **Class EnterpriseDataValidator:** คลาสหลักที่มีฟังก์ชันการทำงานทั้งหมด
- **Integration:** ออกแบบมาเพื่อทำงานร่วมกับ `ElliottWaveDataProcessor` และ `UnifiedEnterpriseLogger`
- **Stateless Design:** แต่ละฟังก์ชันทำงานโดยไม่ขึ้นกับสถานะก่อนหน้า ทำให้ง่ายต่อการทดสอบและบำรุงรักษา

### 🎯 **วัตถุประสงค์**
- **แก้ไขปัญหา `TypeError`:** แก้ไขต้นตอของปัญหา `'<' not supported between instances of 'str' and 'int'` 
  โดยการทำให้แน่ใจว่าข้อมูลทุกคอลัมน์เป็นประเภทตัวเลขก่อนการเปรียบเทียบหรือคำนวณ
- **เพิ่มความเสถียรของ Pipeline:** ทำให้ Pipeline มีความทนทานต่อข้อมูลที่ไม่สมบูรณ์หรือไม่ถูกต้อง
- **สร้างมาตรฐานข้อมูล:** กำหนดมาตรฐานคุณภาพของข้อมูลสำหรับทั้งระบบ

---
"""

import pandas as pd
from typing import Optional

class EnterpriseDataValidator:
    """
    A robust, enterprise-grade validator for financial time-series data.
    Ensures data integrity, correct data types, and handles inconsistencies
    before the data is used in any analytical or ML pipeline.
    """
    def __init__(self, logger):
        """
        Initializes the validator with a logger instance.

        Args:
            logger: An instance of the UnifiedEnterpriseLogger for detailed logging.
        """
        self.logger = logger
        self.component_name = "EnterpriseDataValidator"

    def validate_and_clean(self, df: pd.DataFrame, ohlcv_columns: list) -> Optional[pd.DataFrame]:
        """
        The main validation and cleaning pipeline.

        Args:
            df (pd.DataFrame): The input DataFrame to validate.
            ohlcv_columns (list): A list of column names for Open, High, Low, Close, Volume.

        Returns:
            Optional[pd.DataFrame]: A cleaned and validated DataFrame, or None if validation fails.
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            return None

        self.logger.info(f"Starting enterprise data validation and cleaning for {len(df)} rows.")
        
        initial_rows = len(df)
        
        # 1. Enforce numeric types for critical columns
        df_numeric = self._enforce_numeric_types(df.copy(), ohlcv_columns)
        if df_numeric is None:
            return None # Error already logged

        rows_after_coercion = len(df_numeric)
        if rows_after_coercion < initial_rows:
             self.logger.warning(f"{initial_rows - rows_after_coercion} rows contained non-numeric values and were marked for removal.")

        # 2. Handle missing values
        df_cleaned = self._handle_missing_values(df_numeric, ohlcv_columns)
        if df_cleaned is None:
            return None # Error already logged

        final_rows = len(df_cleaned)
        rows_dropped = initial_rows - final_rows
        
        if rows_dropped > 0:
            self.logger.warning(f"Dropped {rows_dropped} rows due to non-numeric values or missing data in critical columns.")
        
        self.logger.info(f"Data validation and cleaning complete. Final dataset has {final_rows} rows.")
        
        # Log final dtypes for verification
        self.logger.debug(f"Final dtypes:\n{df_cleaned[ohlcv_columns].dtypes.to_string()}")

        return df_cleaned

    def _enforce_numeric_types(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Converts specified columns to numeric types, coercing errors to NaN.
        """
        self.logger.info(f"Enforcing numeric data types for columns: {columns}")
        
        for col in columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.info(f"Column '{col}' is not numeric. Attempting conversion.")
                    original_non_numeric = df[col].apply(type).ne(int).ne(float).sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.logger.info(f"Converted '{col}' to numeric. {original_non_numeric} values were non-numeric.")
                else:
                    self.logger.debug(f"Column '{col}' is already numeric.")
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame. Skipping type enforcement.")
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Detects and removes rows with NaN values in the specified critical columns.
        """
        self.logger.info("Checking for missing values in critical columns...")
        
        initial_rows = len(df)
        
        # Drop rows where any of the critical columns are NaN
        df.dropna(subset=columns, inplace=True)
        
        rows_after_dropna = len(df)
        rows_dropped = initial_rows - rows_after_dropna
        
        if rows_dropped > 0:
            self.logger.warning(f"Removed {rows_dropped} rows with missing data in columns: {columns}")
        else:
            self.logger.info("No missing values found in critical columns.")
            
        return df
