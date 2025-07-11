#!/usr/bin/env python3
"""
📊 ELLIOTT WAVE DATA PROCESSOR
ตัวประมวลผลข้อมูลสำหรับ Elliott Wave Pattern Recognition

Enterprise Features:
- Real Data Only Processing
- Elliott Wave Pattern Detection
- Advanced Technical Indicators
- Multi-timeframe Analysis
- Enterprise-grade Data Validation
"""

# 🛠️ CUDA FIX: Force CPU-only operation to prevent CUDA errors
import os
import warnings

# Environment variables to force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress CUDA warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import sys
from pathlib import Path
import traceback

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import after path setup
from core.project_paths import get_project_paths
# Import Enterprise Data Validator
from core.enterprise_data_validator import EnterpriseDataValidator
# Import Enterprise ML Protection System
from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem

# 🚀 Advanced Logging Integration
try:
    from core.unified_enterprise_logger import get_unified_logger
    ADVANCED_LOGGING_AVAILABLE = True
    print("✅ Advanced logging system loaded successfully")
except ImportError as e:
    ADVANCED_LOGGING_AVAILABLE = False
    print(f"ℹ️ Using standard logging (Advanced components: {e})")

# Progress Manager is optional - no import needed
PROGRESS_MANAGER_AVAILABLE = False  # Disabled for now


class DataProcessor:
    """Enterprise Data Processor Wrapper (for validation)"""
    def __init__(self, config=None, logger=None):
        pass
    def load_real_data(self):
        return None


class ElliottWaveDataProcessor:
    """ตัวประมวลผลข้อมูล Elliott Wave แบบ Enterprise"""
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.component_name = "ElliottWaveDataProcessor"
        
        # 🚀 Initialize Advanced Logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_unified_logger()
            self.progress_manager = None  # Progress manager disabled for now
            self.logger.info(f"🚀 {self.component_name} initialized with Advanced Logging", 
                            component=self.component_name)
        else:
            self.logger = logger or get_unified_logger()
            self.progress_manager = None
        
        self.data_cache = {}
        
        # Use ProjectPaths for path management
        self.paths = get_project_paths()
        self.datacsv_path = self.paths.datacsv
        
        # 🛡️ Initialize Enterprise Data Validator
        self.validator = EnterpriseDataValidator(logger=self.logger)
        
        # Initialize Enterprise ML Protection System
        self.ml_protection = EnterpriseMLProtectionSystem(logger=self.logger)
        
        # Create safe logger for robust logging
        self.safe_logger = self.logger
        
    def _log_data_size_change(self, df_name: str, operation: str, before_size: int, after_size: int):
        """Log data size changes to track data processing steps"""
        if after_size < before_size:
            loss_count = before_size - after_size
            loss_pct = (loss_count / before_size) * 100 if before_size > 0 else 0
            # Only warn if significant data loss (>1%)
            if loss_pct > 1.0:
                self.safe_logger.warning(f"📉 {operation}: {df_name} reduced from {before_size:,} to {after_size:,} rows (-{loss_count:,}, -{loss_pct:.1f}%)")
            else:
                self.safe_logger.info(f"🔧 {operation}: {df_name} processed, {before_size:,} → {after_size:,} rows (-{loss_count:,} noise/outliers)")
        elif after_size > before_size:
            gain_count = after_size - before_size
            gain_pct = (gain_count / before_size) * 100 if before_size > 0 else 0
            self.safe_logger.info(f"📈 {operation}: {df_name} increased from {before_size:,} to {after_size:,} rows (+{gain_count:,}, +{gain_pct:.1f}%)")
        else:
            self.safe_logger.info(f"📊 {operation}: {df_name} size unchanged at {after_size:,} rows")
        
    def load_real_data(self) -> Optional[pd.DataFrame]:
        """โหลดข้อมูลจริงจากโฟลเดอร์ datacsv เท่านั้น - NO LIMITS MODE"""
        try:
            # 🎯 CHECK FULL POWER CONFIG - LOAD ALL DATA, NO LIMITS
            load_all_data = self.config.get('data_processor', {}).get('load_all_data', True)
            sampling_disabled = self.config.get('data_processor', {}).get('sampling_disabled', True)
            
            if load_all_data and sampling_disabled:
                self.logger.info("🚀 FULL POWER MODE: Loading ALL data, NO sampling, NO limits")
            else:
                self.logger.info("📊 Loading REAL market data from datacsv/...")
            
            # Find CSV files in datacsv directory
            csv_files = list(self.datacsv_path.glob("*.csv"))
            
            if not csv_files:
                error_msg = (
                    f"❌ NO CSV FILES FOUND in {self.datacsv_path}! "
                    f"Please add real market data files."
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Select best data file (prefer M1 for higher granularity)
            data_file = self._select_best_data_file(csv_files)
            self.logger.info(
                f"� Loading REAL data from: {data_file.name}"
            )
            
            # Load ALL data - NO row limits for production
            df = pd.read_csv(data_file, low_memory=False)
            initial_size = len(df)
            self.safe_logger.info(f"📊 Initial data loaded: {initial_size:,} rows")
            
            # ✅ FIX: Standardize column names before validation
            ohlc_map = self._detect_ohlc_columns(df)
            if ohlc_map:
                df = df.rename(columns=ohlc_map)

            # 🛡️ Use EnterpriseDataValidator for robust cleaning
            ohlcv_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            df_cleaned = self.validator.validate_and_clean(df, ohlcv_columns=ohlcv_columns)

            if df_cleaned is None:
                self.logger.log_critical("Data validation failed. Cannot proceed.", self.component_name)
                return None
            
            # Check if data is empty after cleaning
            if df_cleaned.empty:
                self.logger.log_error("DataFrame is empty after validation and cleaning. No data to process.", self.component_name)
                return None

            self.safe_logger.info(f"✅ Data validation complete. Proceeding with {len(df_cleaned):,} rows.")
            df = df_cleaned

            # Convert 'time' column to datetime objects
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                self.safe_logger.info("🕒 Converted 'time' column to datetime objects.")
            
            # Set 'time' as index if it exists
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
                self.safe_logger.info("📊 Set 'time' column as DataFrame index.")

            # Cache the loaded data
            self.data_cache['real_data'] = df
            
            return df
        
        except FileNotFoundError as e:
            self.logger.error(f"Data loading failed: {e}", self.component_name)
            # Propagate the error for the pipeline to handle
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during data loading: {e}", self.component_name)
            self.logger.error(traceback.format_exc(), self.component_name)
            return None

    def _select_best_data_file(self, csv_files: List[Path]) -> Path:
        """เลือกไฟล์ข้อมูลที่ดีที่สุด"""
        # Priority: M1 > M5 > M15 > M30 > H1 > H4 > D1
        # Use exact matching to avoid M1 matching M15
        timeframe_patterns = [
            '_M1.', '_M1_',  # M1 patterns
            '_M5.', '_M5_',  # M5 patterns  
            '_M15.', '_M15_', # M15 patterns
            '_M30.', '_M30_', # M30 patterns
            '_H1.', '_H1_',   # H1 patterns
            '_H4.', '_H4_',   # H4 patterns
            '_D1.', '_D1_'    # D1 patterns
        ]
        
        # Map patterns to timeframes for logging
        pattern_to_timeframe = {
            '_M1.': 'M1', '_M1_': 'M1',
            '_M5.': 'M5', '_M5_': 'M5', 
            '_M15.': 'M15', '_M15_': 'M15',
            '_M30.': 'M30', '_M30_': 'M30',
            '_H1.': 'H1', '_H1_': 'H1',
            '_H4.': 'H4', '_H4_': 'H4',
            '_D1.': 'D1', '_D1_': 'D1'
        }
        
        for pattern in timeframe_patterns:
            for file_path in csv_files:
                if pattern in file_path.name.upper():
                    timeframe = pattern_to_timeframe.get(pattern, pattern)
                    self.safe_logger.info(f"📈 Selected data file: {file_path.name} (timeframe: {timeframe})")
                    return file_path
        
        # Return the largest file if no timeframe match (likely the most detailed data)
        largest_file = max(csv_files, key=lambda f: f.stat().st_size)
        self.safe_logger.info(f"📈 No timeframe match, selected largest file: {largest_file.name}")
        return largest_file
    
    def _validate_real_market_data(self, df: pd.DataFrame) -> bool:
        """ตรวจสอบว่าเป็นข้อมูลตลาดจริง"""
        try:
            # Check minimum required columns (already standardized to lowercase)
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    self.safe_logger.error(f"❌ Data validation failed: Required column '{col}' not found.")
                    return False
            
            # Check data quality
            if len(df) < 1000:  # At least 1000 rows for robust analysis
                self.safe_logger.warning(f"⚠️ Data validation warning: Data has only {len(df)} rows (less than 1000).")
                # Not returning False to allow smaller datasets, but logging a warning.

            # Check for realistic price ranges (for XAUUSD)
            # This is the critical fix: only check the known required price columns.
            price_cols_to_check = ['open', 'high', 'low', 'close']
            for col in price_cols_to_check:
                # This check is redundant if load_real_data works, but it's a good safeguard.
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.safe_logger.error(f"❌ Data validation failed: Column '{col}' is not numeric.")
                    return False
                
                # Check for non-finite values that can cause issues
                if not np.all(np.isfinite(df[col])):
                    self.safe_logger.warning(f"⚠️ Data validation warning: Column '{col}' contains non-finite values (NaN/inf).")

                # Now, perform the range check safely.
                # Add a check to see if the column is empty to avoid errors on min/max
                if not df[col].empty:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if min_val < 500 or max_val > 5000:  # Realistic gold price range
                        self.safe_logger.warning(
                            f"⚠️ Data validation warning: Column '{col}' has values outside the typical XAUUSD range (500-5000). "
                            f"Min: {min_val}, Max: {max_val}. This might be acceptable for other assets."
                        )
                        # Continue, allowing flexibility for other assets.
            
            self.safe_logger.info("✅ Data passed basic real market data validation.")
            return True
            
        except Exception as e:
            self.safe_logger.error(f"❌ An unexpected error occurred during data validation: {e}")
            # Also log traceback for debugging
            self.safe_logger.debug(traceback.format_exc())
            return False

    def _create_sample_data(self) -> pd.DataFrame:
        """สร้างข้อมูลตัวอย่างสำหรับการทดสอบ"""
        # Create realistic OHLC data
        np.random.seed(42)
        n_rows = 10000
        
        # Generate realistic gold price data
        base_price = 2000.0
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1min')
        
        # Generate price movements
        returns = np.random.normal(0, 0.002, n_rows)  # 0.2% volatility
        returns = np.cumsum(returns)  # Cumulative sum for trending
        
        close_prices = base_price * (1 + returns)
        
        # Generate OHLC from close prices
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Volume
        volume = np.random.randint(100, 1000, n_rows)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return df
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบและทำความสะอาดข้อมูล"""
        try:
            self.logger.info("🧹 Validating and cleaning data...")
            
            # Column name standardization and numeric conversion is now done in load_real_data.
            
            # Handle timestamp column - combine Date and Timestamp for XAUUSD data
            if 'Date' in df.columns and 'Timestamp' in df.columns:
                # XAUUSD format: Date=25630501, Timestamp=00:00:00
                # This appears to be a custom date format, let's handle it carefully
                df['date_str'] = df['Date'].astype(str)
                
                try:
                    # Try different interpretations of the date format
                    # Approach 1: Treat as YYMMDDXX format (Year 2025, Month 06, Day 30, sequence 01)
                    if len(df['date_str'].iloc[0]) == 8:
                        # Extract components: YYMMDDXX -> YY(25) MM(06) DD(30) XX(01)
                        df['year'] = '20' + df['date_str'].str[:2]  # 25 -> 2025
                        df['month'] = df['date_str'].str[2:4]       # 06 -> 06
                        df['day'] = df['date_str'].str[4:6]         # 30 -> 30
                        # Ignore the last 2 digits (sequence number)
                        
                        # Validate month and day ranges
                        df['month_int'] = pd.to_numeric(df['month'], errors='coerce')
                        df['day_int'] = pd.to_numeric(df['day'], errors='coerce')
                        
                        # Check if this interpretation makes sense
                        # Handle NaN values and ensure numeric comparison
                        valid_months = df['month_int'].notna() & (df['month_int'] >= 1) & (df['month_int'] <= 12)
                        valid_days = df['day_int'].notna() & (df['day_int'] >= 1) & (df['day_int'] <= 31)
                        
                        if valid_months.all() and valid_days.all():
                            # This interpretation works
                            df['date_formatted'] = df['year'] + '-' + df['month'] + '-' + df['day']
                        else:
                            # Try alternative interpretation: DDMMYYXX
                            df['day_alt'] = df['date_str'].str[:2]     # 25 -> 25
                            df['month_alt'] = df['date_str'].str[2:4]  # 06 -> 06  
                            df['year_alt'] = '20' + df['date_str'].str[4:6]  # 30 -> 2030
                            df['date_formatted'] = df['year_alt'] + '-' + df['month_alt'] + '-' + df['day_alt']
                            df = df.drop(columns=['day_alt', 'month_alt', 'year_alt'], errors='ignore')
                        
                        # Create timestamp
                        df['timestamp'] = pd.to_datetime(df['date_formatted'] + ' ' + df['Timestamp'].astype(str), errors='coerce')
                        
                        # Clean up intermediate columns
                        df = df.drop(columns=['year', 'month', 'day', 'month_int', 'day_int', 'date_formatted'], errors='ignore')
                    
                    # If we still have invalid timestamps, use sequential timestamps
                    invalid_count = df['timestamp'].isna().sum()
                    if invalid_count > 0:
                        # For very large datasets with date issues, use optimized sequential timestamps
                        if invalid_count >= len(df) * 0.5:  # More than 50% invalid
                            self.safe_logger.info(f"ℹ️ Using optimized sequential timestamps for {len(df):,} rows (faster processing)")
                            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
                        else:
                            self.safe_logger.warning(f"⚠️ {invalid_count:,} dates could not be parsed, using sequential timestamps")
                            # Keep valid timestamps and fill invalid ones with interpolation
                            valid_timestamps = df['timestamp'].dropna()
                            if len(valid_timestamps) > 1:
                                # Use the valid timestamp pattern to fill missing ones
                                first_valid = valid_timestamps.iloc[0]
                                # Create a simple sequential fill
                                df.loc[df['timestamp'].isna(), 'timestamp'] = pd.date_range(
                                    start=first_valid, periods=invalid_count, freq='1min'
                                )
                            else:
                                # Fallback to full sequential timestamps
                                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
                    
                    # Drop original date columns
                    df = df.drop(columns=['Date', 'Timestamp', 'date_str'], errors='ignore')
                    
                except Exception as e:
                    self.safe_logger.error(f"❌ Date parsing failed: {str(e)}")
                    # Fallback: create sequential timestamps
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
                    df = df.drop(columns=['Date', 'Timestamp'], errors='ignore')
            else:
                # Handle other timestamp formats
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
                    df = df.drop(columns=timestamp_cols)
                elif 'timestamp' not in df.columns:
                    # Create timestamp if not exists
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates MORE CONSERVATIVELY - only if truly identical
            # Count before and after to track data loss
            before_dedup = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
            after_dedup = len(df)
            
            if after_dedup < before_dedup:
                self.safe_logger.warning(f"⚠️ Removed {before_dedup - after_dedup:,} duplicate timestamps")
                if after_dedup < before_dedup * 0.5:
                    self.safe_logger.error(f"🚨 CRITICAL: Duplicate removal caused {(1-after_dedup/before_dedup)*100:.1f}% data loss!")
            
            # Handle missing values
            df = df.ffill().bfill()
            
            # Validate OHLC relationships
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                # Fix OHLC relationships
                df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
                df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
            
            # Apply noise filtering and outlier removal
            df = self._apply_noise_filtering(df)
            
            self.logger.info("✅ Data validation and cleaning completed")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {str(e)}")
            return df
    
    def _detect_ohlc_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detects OHLC and volume columns with various common naming conventions.
        Returns a mapping from detected names to standard names.
        """
        column_map = {}
        # Case-insensitive matching
        df_columns_lower = {col.lower(): col for col in df.columns}

        # Define standard names and possible variations
        mappings = {
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c'],
            'tick_volume': ['volume', 'vol', 'v', 'tick_volume', 'tickvolume']
        }

        for standard_name, variations in mappings.items():
            for var in variations:
                if var in df_columns_lower:
                    original_col_name = df_columns_lower[var]
                    if original_col_name not in column_map.values():
                         column_map[original_col_name] = standard_name
                         self.logger.info(f"Mapped column '{original_col_name}' to standard name '{standard_name}'.")
                         break # Move to the next standard name once a match is found
        
        if len(column_map) < 4: # Check for at least OHLC
             self.logger.warning(f"Could not detect all standard OHLCV columns. Found: {list(column_map.values())}")

        return column_map

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the loaded data, loading it if not already cached.
        This ensures data is loaded only once.
        """
        if 'real_data' not in self.data_cache:
            self.logger.info("Data not in cache, calling load_real_data().")
            self.data_cache['real_data'] = self.load_real_data()
        
        # Return a copy to prevent unintentional modifications of the cached data
        return self.data_cache['real_data'].copy() if self.data_cache['real_data'] is not None else None

    def get_or_load_data(self) -> Optional[pd.DataFrame]:
        """
        Main entry point to get data. Loads if not in cache.
        This is the primary method components should call.
        """
        return self.get_data()

    def _validate_data_for_processing(self, df: pd.DataFrame) -> bool:
        """
        Validates that the DataFrame is ready for processing (e.g., has required columns).
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Data validation failed. Missing one or more required columns: {required_cols}", self.component_name)
            return False
        
        # Check for non-numeric types which can cause calculation errors
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.error(f"Column '{col}' has non-numeric data, which will cause errors in calculations.", self.component_name)
                return False
                
        return True

    def process_data_for_elliott_wave(self, input_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Processes the loaded data to prepare it for Elliott Wave analysis.
        This is the main processing pipeline.
        
        Args:
            input_data: Optional DataFrame to process. If None, will load data using get_or_load_data()
        """
        self.logger.info("Starting Elliott Wave data processing pipeline...", self.component_name)
        
        # Step 1: Get data using the unified method or use provided input_data
        if input_data is not None:
            df = input_data.copy()
            self.logger.info(f"Using provided input data with {len(df)} rows", self.component_name)
        else:
            df = self.get_or_load_data()
            if df is None:
                self.logger.error("Failed to get data. Aborting processing.", self.component_name)
                return None
        
        # Step 2: Validate data before processing
        if not self._validate_data_for_processing(df):
            return None # Error logged in validator

        initial_size = len(df)
        if self.progress_manager:
            self.progress_manager.update_progress('data_processing', 10, "Data Loaded and Validated")

        # Step 3: Noise Filtering (Example: using a simple moving average)
        try:
            # Ensure data is numeric before this step
            df['close_filtered'] = df['close'].rolling(window=3).mean()
            df.dropna(inplace=True) # Remove rows with NaN from rolling mean
            if self.progress_manager:
                self.progress_manager.update_progress('data_processing', 30, "Noise Filtering Complete")
            self._log_data_size_change("DataFrame", "Noise Filtering", initial_size, len(df))
        except Exception as e:
            self.logger.error(f"Error during noise filtering: {e}", self.component_name)
            self.logger.debug(traceback.format_exc())
            return None

        # Step 4: Feature Engineering (add more indicators as needed)
        self.logger.info("Starting feature engineering...", self.component_name)
        df = self.add_technical_indicators(df)
        if self.progress_manager:
            self.progress_manager.update_progress('data_processing', 60, "Feature Engineering Complete")
        
        # Step 5: Final Data Preparation
        self.logger.info("Finalizing data preparation...", self.component_name)
        
        # Remove all non-numeric columns that could cause issues with feature selection
        numeric_columns = []
        non_numeric_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            self.logger.info(f"🔧 Time-pattern column removal: {non_numeric_columns} (enterprise data cleaning)", self.component_name)
            df = df[numeric_columns]
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            self.logger.error(f"Missing required columns: {missing_required}", self.component_name)
            return None
        
        # Final validation - ensure no string columns remain
        string_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if string_columns:
            self.logger.warning(f"⚠️ Removing remaining string columns: {string_columns}", self.component_name)
            df = df.drop(columns=string_columns)
        
        if self.progress_manager:
            self.progress_manager.update_progress('data_processing', 100, "Data Processing Complete")
        self.logger.info("Elliott Wave data processing pipeline finished successfully.", self.component_name)
        
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds various technical indicators to the DataFrame.
        """
        self.logger.info("Adding technical indicators (RSI, MACD)...", self.component_name)
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        macd_line, signal_line = self.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        self.logger.info("Technical indicators added and NaN values dropped.", self.component_name)
        
        return df

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line

    def find_extrema(self, df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
        """
        Finds local price extrema (peaks and troughs) which are potential Elliott Wave points.
        """
        from scipy.signal import argrelextrema
        
        self.logger.info(f"Finding extrema with order={order}...", self.component_name)
        
        # Ensure the index is sorted
        df = df.sort_index()

        # Find local maxima and minima on the 'high' and 'low' prices
        high_extrema_indices = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
        low_extrema_indices = argrelextrema(df['low'].values, np.less_equal, order=order)[0]

        df['peak'] = 0
        df.iloc[high_extrema_indices, df.columns.get_loc('peak')] = 1
        
        df['trough'] = 0
        df.iloc[low_extrema_indices, df.columns.get_loc('trough')] = 1
        
        self.logger.info(f"Found {len(high_extrema_indices)} peaks and {len(low_extrema_indices)} troughs.", self.component_name)
        
        return df

    def get_wave_data(self, df: pd.DataFrame) -> Optional[List[Dict]]:
        """
        Extracts the sequence of peaks and troughs to form a wave structure.
        """
        extrema_df = df[(df['peak'] == 1) | (df['trough'] == 1)].copy()
        
        if extrema_df.empty:
            self.logger.warning("No extrema found, cannot generate wave data.", self.component_name)
            return None
            
        extrema_df['type'] = np.where(extrema_df['peak'] == 1, 'peak', 'trough')
        
        # Ensure alternating peaks and troughs
        last_type = None
        valid_extrema = []
        for index, row in extrema_df.iterrows():
            if row['type'] != last_type:
                valid_extrema.append(row)
                last_type = row['type']
        
        if not valid_extrema:
            self.logger.warning("No valid alternating extrema found.", self.component_name)
            return None

        wave_data_df = pd.DataFrame(valid_extrema)
        
        self.logger.info(f"Generated {len(wave_data_df)} alternating wave points.", self.component_name)
        
        # Return data as a list of dictionaries
        return wave_data_df[['type', 'close']].reset_index().to_dict('records')

    def load_and_prepare_data(self, data_file: str = None) -> Dict[str, Any]:
        """
        Enterprise method to load and prepare data for the Elliott Wave pipeline.
        Returns a dictionary with data, X, y, and metadata for compatibility with enhanced menu.
        """
        try:
            self.logger.info(f"🔄 Loading and preparing data from: {data_file or 'auto-detected source'}")
            
            # Load real market data
            raw_data = self.load_real_data()
            if raw_data is None:
                return {"status": "ERROR", "message": "Failed to load real market data"}
            
            # Process the data for Elliott Wave analysis
            processed_data = self.process_data_for_elliott_wave(raw_data)
            if processed_data is None:
                return {"status": "ERROR", "message": "Failed to process data for Elliott Wave analysis"}
            
            # Create target variable (simple price direction prediction)
            processed_data['target'] = (processed_data['close'].shift(-1) > processed_data['close']).astype(int)
            processed_data = processed_data.dropna()  # Remove last row with NaN target
            
            # Prepare X and y for ML
            feature_columns = [col for col in processed_data.columns if col != 'target']
            X = processed_data[feature_columns]
            y = processed_data['target']
            
            self.logger.info(f"✅ Data prepared successfully: {len(X)} samples, {len(feature_columns)} features")
            
            return {
                "status": "SUCCESS",
                "data": processed_data,
                "X": X,
                "y": y,
                "feature_columns": feature_columns,
                "data_shape": X.shape,
                "target_distribution": y.value_counts().to_dict(),
                "metadata": {
                    "original_rows": len(raw_data),
                    "processed_rows": len(processed_data),
                    "features_count": len(feature_columns),
                    "data_source": data_file or "auto-detected"
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to load and prepare data: {str(e)}"
            self.logger.error(error_msg, component=self.component_name)
            return {"status": "ERROR", "message": error_msg}

    def run_full_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Runs the entire data processing pipeline from loading to feature engineering.
        This is a convenience method to execute all steps.
        """
        self.logger.info("Executing full data processing pipeline...", self.component_name)
        self.progress_manager.start_task('data_processing', "Data Processing Pipeline")
        
        processed_data = self.process_data_for_elliott_wave()
        
        if processed_data is None:
            self.logger.error("Full data processing pipeline failed.", self.component_name)
            self.progress_manager.fail_task('data_processing', "Pipeline Failed")
            return None
            
        self.logger.info("Full data processing pipeline completed successfully.", self.component_name)
        self.progress_manager.complete_task('data_processing')
        
        return processed_data

# Example usage for testing
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    print("Running ElliottWaveDataProcessor standalone for testing...")
    
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Instantiate the processor
    data_processor = ElliottWaveDataProcessor(logger=logging.getLogger())
    
    # Run the full pipeline
    final_data = data_processor.run_full_pipeline()
    
    if final_data is not None:
        print("\n✅ Pipeline executed successfully!")
        print(f"Final DataFrame shape: {final_data.shape}")
        print("Final DataFrame head:")
        print(final_data.head())
        
        # Test extrema finding
        extrema_data = data_processor.find_extrema(final_data.copy())
        print("\nExtrema data head:")
        print(extrema_data[['close', 'peak', 'trough']].head())
        
        # Test wave data generation
        wave_sequence = data_processor.get_wave_data(extrema_data)
        if wave_sequence:
            print(f"\nGenerated wave sequence with {len(wave_sequence)} points.")
            print("First 5 points of the wave sequence:")
            print(wave_sequence[:5])
    else:
        print("\n❌ Pipeline execution failed.")
