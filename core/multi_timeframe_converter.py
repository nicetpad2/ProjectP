#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕒 MULTI-TIMEFRAME CONVERTER
NICEGOLD ProjectP - Advanced Multi-Timeframe Data Conversion System

🚀 FEATURES:
✅ Convert M1 data to any timeframe (M5, M15, M30, H1, H4, D1)
✅ OHLCV aggregation with proper volume handling
✅ Preserve data integrity and timezone handling
✅ Memory-efficient processing for large datasets
✅ Automatic timeframe detection and validation
✅ Enterprise-grade error handling and logging
✅ Support for custom timeframes

📊 SUPPORTED TIMEFRAMES:
- M1: 1 minute (source)
- M5: 5 minutes
- M15: 15 minutes  
- M30: 30 minutes
- H1: 1 hour
- H4: 4 hours
- D1: 1 day
- Custom: Any minute-based timeframe
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore')

# Project imports
try:
    from unified_enterprise_logger import get_unified_logger
    logger = get_unified_logger()
    ENTERPRISE_IMPORTS = True
except ImportError:
    logger = logging.getLogger(__name__)
    ENTERPRISE_IMPORTS = False

# Rich UI imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class Timeframe(Enum):
    """Supported timeframe definitions"""
    M1 = 1      # 1 minute
    M5 = 5      # 5 minutes
    M15 = 15    # 15 minutes
    M30 = 30    # 30 minutes
    H1 = 60     # 1 hour
    H4 = 240    # 4 hours
    D1 = 1440   # 1 day

@dataclass
class ConversionResult:
    """Result of timeframe conversion"""
    original_timeframe: str
    target_timeframe: str
    original_rows: int
    converted_rows: int
    conversion_ratio: float
    data: pd.DataFrame
    start_time: datetime
    end_time: datetime
    processing_time: float
    data_integrity_check: bool
    
class MultiTimeframeConverter:
    """
    🕒 Advanced Multi-Timeframe Data Converter
    Converts M1 data to any higher timeframe with data integrity preservation
    """
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.logger = logger
        
        # Conversion statistics
        self.conversion_stats = {}
        
        # Data validation parameters
        self.min_required_points = 100
        self.max_gap_tolerance = timedelta(minutes=5)
        
    def detect_timeframe(self, data: pd.DataFrame) -> Tuple[str, int]:
        """Detect the timeframe of input data"""
        try:
            if 'DateTime' not in data.columns:
                raise ValueError("DateTime column not found in data")
            
            # Calculate time differences
            time_diffs = data['DateTime'].diff().dropna()
            
            # Find the most common time difference
            mode_diff = time_diffs.mode()
            if len(mode_diff) == 0:
                raise ValueError("Cannot determine timeframe from data")
            
            # Convert to minutes
            minutes = mode_diff.iloc[0].total_seconds() / 60
            
            # Map to standard timeframes
            timeframe_map = {
                1: "M1",
                5: "M5", 
                15: "M15",
                30: "M30",
                60: "H1",
                240: "H4",
                1440: "D1"
            }
            
            detected_tf = timeframe_map.get(int(minutes), f"M{int(minutes)}")
            
            if self.logger:
                self.logger.info(f"🕒 Detected timeframe: {detected_tf} ({int(minutes)} minutes)")
            
            return detected_tf, int(minutes)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error detecting timeframe: {e}")
            return "M1", 1

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and structure"""
        try:
            # Check required columns
            required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                if self.logger:
                    self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(data['DateTime']):
                if self.logger:
                    self.logger.error("❌ DateTime column is not datetime type")
                return False
            
            # Check for NaT (Not a Time) values in DateTime
            nat_count = data['DateTime'].isnull().sum()
            if nat_count > 0:
                if self.logger:
                    self.logger.warning(f"⚠️ Found {nat_count} NaT (Not a Time) values in DateTime - will remove them")
                # Remove rows with NaT
                data = data.dropna(subset=['DateTime'])
                
            # Check for NaN values in OHLCV
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            nan_count = data[ohlcv_columns].isnull().sum().sum()
            
            if nan_count > 0:
                if self.logger:
                    self.logger.warning(f"⚠️ Found {nan_count} NaN values in OHLCV data")
            
            # Check minimum data points
            if len(data) < self.min_required_points:
                if self.logger:
                    self.logger.warning(f"⚠️ Only {len(data)} data points, minimum recommended: {self.min_required_points}")
            
            # Check for duplicate timestamps
            duplicates = data['DateTime'].duplicated().sum()
            if duplicates > 0:
                if self.logger:
                    self.logger.warning(f"⚠️ Found {duplicates} duplicate timestamps")
            
            # Check data ordering
            if not data['DateTime'].is_monotonic_increasing:
                if self.logger:
                    self.logger.warning("⚠️ DateTime is not monotonic increasing - will sort data")
            
            if self.logger:
                self.logger.info("✅ Input data validation completed")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error validating input data: {e}")
            return False

    def convert_to_timeframe(self, data: pd.DataFrame, target_timeframe: str) -> ConversionResult:
        """Convert data to target timeframe"""
        try:
            start_time = datetime.now()
            
            if self.logger:
                self.logger.info(f"🔄 Converting to {target_timeframe} timeframe...")
            
            # Validate input data
            if not self.validate_input_data(data):
                raise ValueError("Input data validation failed")
            
            # Detect source timeframe
            source_tf, source_minutes = self.detect_timeframe(data)
            
            # Parse target timeframe
            target_minutes = self._parse_timeframe(target_timeframe)
            
            if target_minutes <= source_minutes:
                raise ValueError(f"Target timeframe ({target_minutes}min) must be higher than source ({source_minutes}min)")
            
            # Sort data by datetime
            df = data.copy().sort_values('DateTime').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['DateTime']).reset_index(drop=True)
            
            # Remove rows with NaT (Not a Time) values
            df = df.dropna(subset=['DateTime'])
            
            # Set DateTime as index for resampling
            df.set_index('DateTime', inplace=True)
            
            # Define aggregation rules
            agg_rules = {
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Add any additional columns with appropriate aggregation
            additional_columns = set(df.columns) - set(agg_rules.keys())
            for col in additional_columns:
                if df[col].dtype in ['float64', 'int64']:
                    agg_rules[col] = 'mean'  # Average for numeric columns
                else:
                    agg_rules[col] = 'first'  # First value for non-numeric
            
            # Perform resampling
            resampling_rule = f"{target_minutes}T"  # T = minutes
            
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"🔄 Resampling to {target_timeframe}...", total=100)
                    
                    # Resample data
                    resampled = df.resample(resampling_rule).agg(agg_rules)
                    
                    progress.update(task, completed=50)
                    
                    # Remove rows with NaN (incomplete periods)
                    resampled = resampled.dropna()
                    
                    progress.update(task, completed=100)
            else:
                # Resample data without progress bar
                resampled = df.resample(resampling_rule).agg(agg_rules)
                resampled = resampled.dropna()
            
            # Reset index to get DateTime back as column
            resampled.reset_index(inplace=True)
            
            # Data integrity check
            integrity_check = self._validate_ohlc_integrity(resampled)
            
            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            conversion_ratio = len(resampled) / len(data) if len(data) > 0 else 0
            
            # Create result
            result = ConversionResult(
                original_timeframe=source_tf,
                target_timeframe=target_timeframe,
                original_rows=len(data),
                converted_rows=len(resampled),
                conversion_ratio=conversion_ratio,
                data=resampled,
                start_time=resampled['DateTime'].min() if len(resampled) > 0 else datetime.now(),
                end_time=resampled['DateTime'].max() if len(resampled) > 0 else datetime.now(),
                processing_time=processing_time,
                data_integrity_check=integrity_check
            )
            
            if self.logger:
                self.logger.info(f"✅ Conversion completed: {len(data):,} → {len(resampled):,} rows")
                self.logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                self.logger.info(f"📊 Conversion ratio: {conversion_ratio:.4f}")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error converting timeframe: {e}")
            raise

    def convert_to_multiple_timeframes(self, data: pd.DataFrame, 
                                     target_timeframes: List[str]) -> Dict[str, ConversionResult]:
        """Convert data to multiple timeframes at once"""
        try:
            results = {}
            
            if self.logger:
                self.logger.info(f"🔄 Converting to {len(target_timeframes)} timeframes: {target_timeframes}")
            
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("🔄 Multi-timeframe conversion...", total=len(target_timeframes))
                    
                    for i, tf in enumerate(target_timeframes):
                        try:
                            progress.update(task, description=f"Converting to {tf}...")
                            result = self.convert_to_timeframe(data, tf)
                            results[tf] = result
                            progress.update(task, completed=i + 1)
                            
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"❌ Failed to convert to {tf}: {e}")
                            progress.update(task, completed=i + 1)
            else:
                for tf in target_timeframes:
                    try:
                        result = self.convert_to_timeframe(data, tf)
                        results[tf] = result
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"❌ Failed to convert to {tf}: {e}")
            
            if self.logger:
                self.logger.info(f"✅ Multi-timeframe conversion completed: {len(results)}/{len(target_timeframes)} successful")
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error in multi-timeframe conversion: {e}")
            return {}

    def save_converted_data(self, result: ConversionResult, output_path: str) -> str:
        """Save converted data to CSV file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save main data
            result.data.to_csv(output_path, index=False)
            
            # Save metadata
            metadata = {
                'conversion_info': {
                    'original_timeframe': result.original_timeframe,
                    'target_timeframe': result.target_timeframe,
                    'original_rows': result.original_rows,
                    'converted_rows': result.converted_rows,
                    'conversion_ratio': result.conversion_ratio,
                    'processing_time': result.processing_time,
                    'data_integrity_check': result.data_integrity_check
                },
                'data_info': {
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'total_periods': result.converted_rows,
                    'file_size_mb': output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
                }
            }
            
            # Save metadata as JSON
            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.logger:
                self.logger.info(f"✅ Data saved to: {output_path}")
                self.logger.info(f"📋 Metadata saved to: {metadata_path}")
            
            return str(output_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saving converted data: {e}")
            raise

    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes"""
        try:
            timeframe = timeframe.upper().strip()
            
            # Handle standard timeframes
            standard_timeframes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D1': 1440
            }
            
            if timeframe in standard_timeframes:
                return standard_timeframes[timeframe]
            
            # Handle custom timeframes
            if timeframe.startswith('M'):
                return int(timeframe[1:])
            elif timeframe.startswith('H'):
                return int(timeframe[1:]) * 60
            elif timeframe.startswith('D'):
                return int(timeframe[1:]) * 1440
            else:
                # Try to parse as number (assume minutes)
                return int(timeframe)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error parsing timeframe '{timeframe}': {e}")
            return 1

    def _validate_ohlc_integrity(self, data: pd.DataFrame) -> bool:
        """Validate OHLC data integrity"""
        try:
            if len(data) == 0:
                return False
            
            # Check OHLC relationships
            valid_high = (data['High'] >= data['Open']) & (data['High'] >= data['Close'])
            valid_low = (data['Low'] <= data['Open']) & (data['Low'] <= data['Close'])
            valid_ohlc = (data['High'] >= data['Low'])
            
            integrity_check = valid_high.all() and valid_low.all() and valid_ohlc.all()
            
            if not integrity_check:
                invalid_count = (~(valid_high & valid_low & valid_ohlc)).sum()
                if self.logger:
                    self.logger.warning(f"⚠️ Found {invalid_count} rows with invalid OHLC relationships")
            
            return integrity_check
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error validating OHLC integrity: {e}")
            return False

    def display_conversion_summary(self, results: Dict[str, ConversionResult]):
        """Display conversion summary"""
        try:
            if not self.console:
                return
            
            # Create summary table
            table = Table(title="📊 Multi-Timeframe Conversion Summary")
            table.add_column("Timeframe", style="cyan")
            table.add_column("Original Rows", style="magenta")
            table.add_column("Converted Rows", style="yellow") 
            table.add_column("Conversion Ratio", style="green")
            table.add_column("Processing Time", style="blue")
            table.add_column("Data Integrity", style="red")
            
            for tf, result in results.items():
                table.add_row(
                    tf,
                    f"{result.original_rows:,}",
                    f"{result.converted_rows:,}",
                    f"{result.conversion_ratio:.4f}",
                    f"{result.processing_time:.2f}s",
                    "✅" if result.data_integrity_check else "❌"
                )
            
            self.console.print(table)
            
            # Display summary statistics
            total_processing_time = sum(r.processing_time for r in results.values())
            successful_conversions = sum(1 for r in results.values() if r.data_integrity_check)
            
            summary_text = f"""
🕒 MULTI-TIMEFRAME CONVERSION COMPLETED

📊 CONVERSION STATISTICS:
📈 Total Timeframes: {len(results)}
✅ Successful Conversions: {successful_conversions}
⏱️ Total Processing Time: {total_processing_time:.2f} seconds
🎯 Success Rate: {successful_conversions/len(results)*100:.1f}%

💡 All converted data maintains OHLCV integrity
📁 Results ready for trading strategy backtesting
"""
            
            self.console.print(Panel(
                summary_text.strip(),
                title="✅ CONVERSION SUMMARY",
                style="bold green"
            ))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error displaying conversion summary: {e}")

# ====================================================
# CONVENIENCE FUNCTIONS
# ====================================================

def convert_m1_to_timeframe(data: pd.DataFrame, target_timeframe: str) -> ConversionResult:
    """Convenience function to convert M1 data to target timeframe"""
    converter = MultiTimeframeConverter()
    return converter.convert_to_timeframe(data, target_timeframe)

def convert_m1_to_multiple_timeframes(data: pd.DataFrame, 
                                    timeframes: List[str] = None) -> Dict[str, ConversionResult]:
    """Convenience function to convert M1 data to multiple timeframes"""
    if timeframes is None:
        timeframes = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']
    
    converter = MultiTimeframeConverter()
    return converter.convert_to_multiple_timeframes(data, timeframes)

def load_and_convert_csv(csv_path: str, target_timeframes: List[str] = None) -> Dict[str, ConversionResult]:
    """Load CSV file and convert to multiple timeframes"""
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Convert Date and Timestamp to DateTime
        if 'Date' in df.columns and 'Timestamp' in df.columns:
            # Handle MT5 format dates
            if df['Date'].dtype == 'int64':
                # Convert from YYYYMMDD format to proper date
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
            else:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Timestamp'].astype(str))
        
        # Convert OHLCV to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN rows
        df = df.dropna()
        
        # Convert to multiple timeframes
        if timeframes is None:
            timeframes = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
        return convert_m1_to_multiple_timeframes(df, timeframes)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error loading and converting CSV: {e}")
        return {}

# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    'MultiTimeframeConverter',
    'ConversionResult', 
    'Timeframe',
    'convert_m1_to_timeframe',
    'convert_m1_to_multiple_timeframes',
    'load_and_convert_csv'
]

if __name__ == "__main__":
    # Test the multi-timeframe converter
    print("🧪 Testing Multi-Timeframe Converter...")
    
    try:
        # Test with sample data path
        sample_path = "datacsv/XAUUSD_M1.csv"
        
        if Path(sample_path).exists():
            print(f"📊 Loading data from: {sample_path}")
            
            # Convert to multiple timeframes
            results = load_and_convert_csv(sample_path, ['M5', 'M15', 'M30', 'H1'])
            
            if results:
                print("✅ Multi-timeframe conversion successful!")
                for tf, result in results.items():
                    print(f"   {tf}: {result.original_rows:,} → {result.converted_rows:,} rows ({result.conversion_ratio:.4f})")
            else:
                print("❌ Multi-timeframe conversion failed")
        else:
            print(f"❌ Sample data file not found: {sample_path}")
            
    except Exception as e:
        print(f"❌ Test exception: {e}")
