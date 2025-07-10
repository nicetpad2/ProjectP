#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 PIPELINE DATA CONTAINER
Standardized container for pipeline data flow with comprehensive metadata tracking

Features:
- Consistent data structure passing between pipeline steps
- Comprehensive metadata tracking
- Error and warning accumulation
- Processing history logging
- Data validation and integrity checks
- Performance metrics collection
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class PipelineDataContainer:
    """Standardized container for pipeline data flow"""
    
    def __init__(self, data: Optional[Any] = None, metadata: Optional[Dict] = None):
        """
        Initialize pipeline data container
        
        Args:
            data: Initial data for the pipeline
            metadata: Initial metadata dictionary
        """
        # Core data storage
        self.data = data
        self.metadata = metadata or {}
        
        # Pipeline state
        self.status = 'initialized'
        self.errors = []
        self.warnings = []
        self.processing_history = []
        
        # Performance tracking
        self.performance_metrics = defaultdict(dict)
        self.timing_info = defaultdict(float)
        
        # Data validation
        self.data_integrity = {
            'initial_shape': self._get_data_shape(data),
            'initial_type': type(data).__name__,
            'validation_checks': []
        }
        
        # Creation timestamp
        self.created_at = datetime.now()
        self.last_modified = self.created_at
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize default metadata"""
        if 'pipeline_id' not in self.metadata:
            self.metadata['pipeline_id'] = f"pipeline_{int(time.time())}"
        
        if 'session_id' not in self.metadata:
            self.metadata['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata['creation_timestamp'] = self.created_at.isoformat()
        self.metadata['data_type'] = self.data_integrity['initial_type']
        self.metadata['initial_shape'] = self.data_integrity['initial_shape']
    
    def _get_data_shape(self, data: Any) -> Optional[tuple]:
        """Get data shape if available"""
        if hasattr(data, 'shape'):
            return data.shape
        elif hasattr(data, '__len__'):
            return (len(data),)
        return None
    
    def add_step_result(self, step_name: str, result_data: Any, 
                       step_metadata: Optional[Dict] = None,
                       execution_time: Optional[float] = None,
                       step_status: str = 'success') -> 'PipelineDataContainer':
        """
        Add result from a pipeline step
        
        Args:
            step_name: Name of the pipeline step
            result_data: Data result from the step
            step_metadata: Additional metadata from the step
            execution_time: Step execution time in seconds
            step_status: Status of the step (success/warning/error)
        """
        # Update primary data
        self.data = result_data
        
        # Update metadata
        if step_metadata:
            self.metadata.update(step_metadata)
        
        # Record processing history
        step_record = {
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'status': step_status,
            'data_shape': self._get_data_shape(result_data),
            'data_type': type(result_data).__name__,
            'memory_usage': self._get_memory_usage(result_data)
        }
        
        self.processing_history.append(step_record)
        
        # Update performance metrics
        if execution_time:
            self.timing_info[step_name] = execution_time
        
        # Update data integrity tracking
        self.data_integrity['validation_checks'].append({
            'step': step_name,
            'shape': step_record['data_shape'],
            'type': step_record['data_type'],
            'timestamp': step_record['timestamp']
        })
        
        # Update container state
        self.last_modified = datetime.now()
        self.status = step_status
        
        return self
    
    def _get_memory_usage(self, data: Any) -> Optional[float]:
        """Get approximate memory usage of data"""
        try:
            if hasattr(data, 'memory_usage'):
                # For pandas DataFrames
                return data.memory_usage(deep=True).sum()
            elif hasattr(data, 'nbytes'):
                # For numpy arrays
                return data.nbytes
            elif hasattr(data, '__sizeof__'):
                return data.__sizeof__()
            return None
        except Exception:
            return None
    
    def get_data(self, step_name: Optional[str] = None) -> Any:
        """
        Get data, optionally from specific step
        
        Args:
            step_name: Optional step name to get data from processing history
        """
        if step_name:
            # Find data from specific step in history
            for record in reversed(self.processing_history):
                if record['step_name'] == step_name:
                    # Note: This is a simplified implementation
                    # In practice, you might want to store data snapshots
                    return self.data
            return None
        
        return self.data
    
    def add_error(self, error_message: str, step_name: Optional[str] = None, 
                  exception: Optional[Exception] = None):
        """Add error to container"""
        error_record = {
            'message': error_message,
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'exception_type': type(exception).__name__ if exception else None,
            'exception_details': str(exception) if exception else None
        }
        
        self.errors.append(error_record)
        self.status = 'error'
        self.last_modified = datetime.now()
    
    def add_warning(self, warning_message: str, step_name: Optional[str] = None):
        """Add warning to container"""
        warning_record = {
            'message': warning_message,
            'step_name': step_name,
            'timestamp': datetime.now().isoformat()
        }
        
        self.warnings.append(warning_record)
        self.last_modified = datetime.now()
    
    def has_data(self) -> bool:
        """Check if container has data"""
        return self.data is not None
    
    def is_empty(self) -> bool:
        """Check if data is empty"""
        if not self.has_data():
            return True
        
        if hasattr(self.data, 'empty'):
            return self.data.empty
        elif hasattr(self.data, '__len__'):
            return len(self.data) == 0
        
        return False
    
    def validate_data(self, expected_type: Optional[type] = None, 
                     expected_shape: Optional[tuple] = None,
                     min_rows: Optional[int] = None) -> bool:
        """
        Validate data against expected criteria
        
        Args:
            expected_type: Expected data type
            expected_shape: Expected data shape
            min_rows: Minimum number of rows required
        """
        if not self.has_data():
            self.add_error("No data available for validation")
            return False
        
        # Type validation
        if expected_type and not isinstance(self.data, expected_type):
            self.add_error(f"Data type mismatch: expected {expected_type.__name__}, got {type(self.data).__name__}")
            return False
        
        # Shape validation
        if expected_shape and hasattr(self.data, 'shape'):
            if self.data.shape != expected_shape:
                self.add_error(f"Shape mismatch: expected {expected_shape}, got {self.data.shape}")
                return False
        
        # Minimum rows validation
        if min_rows and hasattr(self.data, '__len__'):
            if len(self.data) < min_rows:
                self.add_error(f"Insufficient data: expected at least {min_rows} rows, got {len(self.data)}")
                return False
        
        return True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = sum(self.timing_info.values())
        
        return {
            'total_execution_time': total_time,
            'step_timings': dict(self.timing_info),
            'step_count': len(self.processing_history),
            'average_step_time': total_time / max(len(self.processing_history), 1),
            'slowest_step': max(self.timing_info.items(), key=lambda x: x[1]) if self.timing_info else None,
            'fastest_step': min(self.timing_info.items(), key=lambda x: x[1]) if self.timing_info else None
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary"""
        return {
            'status': self.status,
            'has_data': self.has_data(),
            'is_empty': self.is_empty(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'processing_steps': len(self.processing_history),
            'last_modified': self.last_modified.isoformat(),
            'uptime': (datetime.now() - self.created_at).total_seconds()
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get data summary"""
        data_summary = {
            'has_data': self.has_data(),
            'data_type': type(self.data).__name__ if self.data is not None else None,
            'data_shape': self._get_data_shape(self.data),
            'memory_usage': self._get_memory_usage(self.data)
        }
        
        # Add specific summaries for known data types
        if isinstance(self.data, pd.DataFrame):
            data_summary.update({
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'null_counts': self.data.isnull().sum().to_dict(),
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns)
            })
        elif isinstance(self.data, np.ndarray):
            data_summary.update({
                'dtype': str(self.data.dtype),
                'ndim': self.data.ndim,
                'size': self.data.size
            })
        
        return data_summary
    
    def create_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        checkpoint = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'data_summary': self.get_data_summary(),
            'status_summary': self.get_status_summary(),
            'performance_summary': self.get_performance_summary(),
            'metadata': self.metadata.copy(),
            'processing_history': self.processing_history.copy()
        }
        
        return checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary representation"""
        return {
            'metadata': self.metadata,
            'status': self.status,
            'errors': self.errors,
            'warnings': self.warnings,
            'processing_history': self.processing_history,
            'performance_metrics': dict(self.performance_metrics),
            'timing_info': dict(self.timing_info),
            'data_integrity': self.data_integrity,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'data_summary': self.get_data_summary(),
            'status_summary': self.get_status_summary(),
            'performance_summary': self.get_performance_summary()
        }
    
    def __str__(self):
        return f"PipelineDataContainer(status={self.status}, steps={len(self.processing_history)}, errors={len(self.errors)})"
    
    def __repr__(self):
        return f"<PipelineDataContainer: {self.status}, {len(self.processing_history)} steps>"


def create_pipeline_container(data: Any = None, **kwargs) -> PipelineDataContainer:
    """Factory function to create pipeline data container"""
    return PipelineDataContainer(data=data, metadata=kwargs)


# Utility functions for pipeline data management
def safe_extract_data(container_or_data: Union[PipelineDataContainer, Any], 
                     key: str = 'data', fallback: Any = None) -> Any:
    """
    Safely extract data from container or return data directly
    
    Args:
        container_or_data: PipelineDataContainer or direct data
        key: Key to extract from container (unused if not container)
        fallback: Fallback value if extraction fails
    """
    if isinstance(container_or_data, PipelineDataContainer):
        return container_or_data.get_data() if container_or_data.has_data() else fallback
    elif isinstance(container_or_data, dict):
        return container_or_data.get(key, fallback)
    elif hasattr(container_or_data, key):
        return getattr(container_or_data, key, fallback)
    else:
        return container_or_data if container_or_data is not None else fallback


def validate_pipeline_data(container: PipelineDataContainer, 
                          step_name: str, 
                          required_type: Optional[type] = None,
                          min_rows: Optional[int] = None) -> bool:
    """
    Validate pipeline data for a step
    
    Args:
        container: Pipeline data container
        step_name: Name of the step performing validation
        required_type: Required data type
        min_rows: Minimum number of rows
    """
    try:
        if not container.has_data():
            container.add_error(f"No data available for step: {step_name}", step_name)
            return False
        
        if container.is_empty():
            container.add_error(f"Empty data for step: {step_name}", step_name)
            return False
        
        # Perform additional validations
        return container.validate_data(
            expected_type=required_type,
            min_rows=min_rows
        )
        
    except Exception as e:
        container.add_error(f"Validation failed for step {step_name}: {str(e)}", step_name, e)
        return False


# Export main classes and functions
__all__ = [
    'PipelineDataContainer',
    'create_pipeline_container',
    'safe_extract_data',
    'validate_pipeline_data'
]
