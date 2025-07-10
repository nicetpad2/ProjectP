#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE MODEL MANAGER
Enterprise-Grade Model Management System for Production Trading Systems

🎯 Features:
✅ Enterprise Model Versioning & Metadata
✅ Model Validation & Performance Tracking
✅ Production-Ready Model Deployment
✅ Automated Model Backup & Recovery
✅ Model Performance Monitoring
✅ Enterprise Security & Compliance
✅ Model Lifecycle Management
✅ Cross-Platform Model Compatibility

วันที่: 7 กรกฎาคม 2025
"""

import os
import sys
import json
import pickle
import joblib
import hashlib
import shutil
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Add project root
sys.path.append(str(Path(__file__).parent.parent))
from core.project_paths import get_project_paths

# 🚀 ENTERPRISE UNIFIED LOGGER INTEGRATION
try:
    from core.unified_enterprise_logger import get_unified_logger
    ENTERPRISE_LOGGER_AVAILABLE = True
except ImportError:
    ENTERPRISE_LOGGER_AVAILABLE = False
    print("⚠️ Enterprise Unified Logger not available")

class ModelStatus(Enum):
    """Model Status Enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class ModelType(Enum):
    """Model Type Enumeration"""
    CNN_LSTM = "cnn_lstm"
    DQN_AGENT = "dqn_agent"
    FEATURE_SELECTOR = "feature_selector"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"

@dataclass
class ModelMetadata:
    """Enterprise Model Metadata Structure"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    file_path: str
    file_size: int
    file_hash: str
    
    # Performance Metrics
    performance_metrics: Dict[str, float]
    validation_score: float
    training_duration: float
    
    # Training Configuration
    training_config: Dict[str, Any]
    feature_count: int
    data_samples: int
    
    # Business Information
    business_purpose: str
    deployment_target: str
    risk_level: str
    compliance_status: str
    
    # Technical Information
    dependencies: List[str]
    python_version: str
    framework_version: str
    
    # Backup & Recovery
    backup_path: Optional[str] = None
    recovery_tested: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        # Convert enum values to strings
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert string dates back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        # Convert string enums back to enums
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        return cls(**data)

class EnterpriseModelManager:
    """
    🏢 Enterprise Model Manager
    Production-Grade Model Management System
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """Initialize Enterprise Model Manager"""
        self.config = config or {}
        
        # Initialize Enterprise Logger if available
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.logger = logger or get_unified_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
        
        # Get project paths
        # self.paths = get_project_paths()  # Temporarily disabled for testing
        
        # Setup directories
        # self._setup_directories()  # Temporarily disabled for testing
        
        # Initialize database
        # self._initialize_database()  # Temporarily disabled for testing
        
        # Model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        
        # Load existing models
        # self._load_model_registry()  # Temporarily disabled for testing
        
        if ENTERPRISE_LOGGER_AVAILABLE:
            self.logger.info("🏢 Enterprise Model Manager initialized with Enterprise Logger")
        else:
            self.logger.info("🏢 Enterprise Model Manager initialized")
    
    def _setup_directories(self):
        """Setup enterprise directory structure"""
        try:
            # Main model directories
            self.models_dir = self.paths.models
            self.backups_dir = self.models_dir / "backups"
            self.archive_dir = self.models_dir / "archive"
            self.metadata_dir = self.models_dir / "metadata"
            self.validation_dir = self.models_dir / "validation"
            self.deployment_dir = self.models_dir / "deployment"
            
            # Create directories
            for directory in [self.models_dir, self.backups_dir, self.archive_dir, 
                            self.metadata_dir, self.validation_dir, self.deployment_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("✅ Enterprise model directories created")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup directories: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize SQLite database for model tracking"""
        try:
            self.db_path = self.metadata_dir / "model_registry.db"
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create model registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_hash TEXT NOT NULL,
                    performance_metrics TEXT,
                    validation_score REAL,
                    training_duration REAL,
                    training_config TEXT,
                    feature_count INTEGER,
                    data_samples INTEGER,
                    business_purpose TEXT,
                    deployment_target TEXT,
                    risk_level TEXT,
                    compliance_status TEXT,
                    dependencies TEXT,
                    python_version TEXT,
                    framework_version TEXT,
                    backup_path TEXT,
                    recovery_tested INTEGER DEFAULT 0
                )
            ''')
            
            # Create performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    dataset_name TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_registry (model_id)
                )
            ''')
            
            # Create deployment history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    deployment_time TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployment_status TEXT NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_registry (model_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Enterprise model database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def register_model(self, 
                      model: Any,
                      model_name: str,
                      model_type: ModelType,
                      performance_metrics: Dict[str, float],
                      training_config: Dict[str, Any],
                      business_purpose: str = "Elliott Wave Trading",
                      deployment_target: str = "Production",
                      risk_level: str = "Medium") -> str:
        """
        Register a new model in the enterprise registry
        
        Args:
            model: Trained model object
            model_name: Human-readable model name
            model_type: Type of model (CNN_LSTM, DQN_AGENT, etc.)
            performance_metrics: Dictionary of performance metrics
            training_config: Training configuration used
            business_purpose: Business purpose of the model
            deployment_target: Target deployment environment
            risk_level: Risk assessment (Low/Medium/High)
            
        Returns:
            model_id: Unique model identifier
        """
        if ENTERPRISE_LOGGER_AVAILABLE:
            with menu1_step_context(Menu1Step.MODEL_REGISTRATION):
                return self._register_model_internal(
                    model, model_name, model_type, performance_metrics, 
                    training_config, business_purpose, deployment_target, risk_level
                )
        else:
            return self._register_model_internal(
                model, model_name, model_type, performance_metrics, 
                training_config, business_purpose, deployment_target, risk_level
            )
    
    def _register_model_internal(self, model, model_name, model_type, performance_metrics, 
                                training_config, business_purpose, deployment_target, risk_level):
        """Internal model registration logic"""
        try:
            # Generate unique model ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_type.value}_{model_name}_{timestamp}"
            
            # Save model to file
            model_filename = f"{model_id}.joblib"
            model_path = self.models_dir / model_filename
            
            # Save model using joblib for best compatibility
            joblib.dump(model, model_path)
            
            # Calculate file properties
            file_size = model_path.stat().st_size
            file_hash = self._calculate_file_hash(model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                status=ModelStatus.TRAINED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_path=str(model_path),
                file_size=file_size,
                file_hash=file_hash,
                performance_metrics=performance_metrics,
                validation_score=performance_metrics.get('auc', performance_metrics.get('accuracy', 0.0)),
                training_duration=training_config.get('training_duration', 0.0),
                training_config=training_config,
                feature_count=training_config.get('feature_count', 0),
                data_samples=training_config.get('data_samples', 0),
                business_purpose=business_purpose,
                deployment_target=deployment_target,
                risk_level=risk_level,
                compliance_status="Pending",
                dependencies=self._get_current_dependencies(),
                python_version=sys.version,
                framework_version=self._get_framework_versions()
            )
            
            # Save metadata to database
            self._save_model_to_database(metadata)
            
            # Save metadata to JSON file
            metadata_file = self.metadata_dir / f"{model_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Create backup
            self._create_model_backup(model_id)
            
            # Add to registry
            self.model_registry[model_id] = metadata
            
            self.logger.info(f"✅ Model registered: {model_id}")
            self.logger.info(f"   📁 File: {model_filename} ({file_size:,} bytes)")
            self.logger.info(f"   📊 Performance: {performance_metrics}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def validate_model(self, 
                      model_id: str,
                      validation_data: pd.DataFrame,
                      validation_target: pd.Series,
                      min_performance_threshold: float = 0.70) -> Dict[str, Any]:
        """
        Validate model performance for enterprise compliance
        
        Args:
            model_id: Model identifier
            validation_data: Validation dataset
            validation_target: Validation targets
            min_performance_threshold: Minimum required performance
            
        Returns:
            Validation results dictionary
        """
        if ENTERPRISE_LOGGER_AVAILABLE:
            with menu1_step_context(Menu1Step.MODEL_VALIDATION):
                return self._validate_model_internal(
                    model_id, validation_data, validation_target, min_performance_threshold
                )
        else:
            return self._validate_model_internal(
                model_id, validation_data, validation_target, min_performance_threshold
            )
    
    def _validate_model_internal(self, model_id, validation_data, validation_target, min_performance_threshold):
        """Internal model validation logic"""
        try:
            self.logger.info(f"🔍 Validating model: {model_id}")
            
            # Load model
            model = self.load_model(model_id)
            metadata = self.model_registry[model_id]
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                predictions_proba = model.predict_proba(validation_data)[:, 1]
                predictions = (predictions_proba > 0.5).astype(int)
            else:
                predictions = model.predict(validation_data)
                predictions_proba = predictions
            
            # Calculate performance metrics
            validation_metrics = {
                'accuracy': accuracy_score(validation_target, predictions),
                'precision': precision_score(validation_target, predictions, average='weighted', zero_division=0),
                'recall': recall_score(validation_target, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(validation_target, predictions, average='weighted', zero_division=0),
                'validation_samples': len(validation_data)
            }
            
            # Calculate AUC if possible
            try:
                validation_metrics['auc'] = roc_auc_score(validation_target, predictions_proba)
            except Exception:
                validation_metrics['auc'] = 0.0
            
            # Determine validation status
            primary_metric = validation_metrics.get('auc', validation_metrics['accuracy'])
            validation_passed = primary_metric >= min_performance_threshold
            
            validation_result = {
                'model_id': model_id,
                'validation_time': datetime.now().isoformat(),
                'metrics': validation_metrics,
                'threshold': min_performance_threshold,
                'passed': validation_passed,
                'primary_metric': primary_metric,
                'validation_samples': len(validation_data)
            }
            
            # Update model status
            if validation_passed:
                metadata.status = ModelStatus.VALIDATED
                metadata.compliance_status = "Enterprise Compliant"
                self.logger.info(f"✅ Model validation passed: {primary_metric:.4f} >= {min_performance_threshold}")
            else:
                metadata.compliance_status = "Below Threshold"
                self.logger.warning(f"⚠️ Model validation failed: {primary_metric:.4f} < {min_performance_threshold}")
            
            # Update metadata
            metadata.updated_at = datetime.now()
            self._update_model_in_database(metadata)
            
            # Save validation results
            validation_file = self.validation_dir / f"{model_id}_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_result, f, indent=2)
            
            # Record performance history
            self._record_performance_history(model_id, validation_metrics, "validation")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            raise
    
    def deploy_model(self, 
                    model_id: str,
                    environment: str = "production",
                    deployment_notes: str = "") -> Dict[str, Any]:
        """
        Deploy model to production environment
        
        Args:
            model_id: Model identifier
            environment: Deployment environment
            deployment_notes: Optional deployment notes
            
        Returns:
            Deployment result dictionary
        """
        if ENTERPRISE_LOGGER_AVAILABLE:
            with menu1_step_context(Menu1Step.MODEL_DEPLOYMENT):
                return self._deploy_model_internal(model_id, environment, deployment_notes)
        else:
            return self._deploy_model_internal(model_id, environment, deployment_notes)
    
    def _deploy_model_internal(self, model_id, environment, deployment_notes):
        """Internal model deployment logic"""
        try:
            self.logger.info(f"🚀 Deploying model: {model_id} to {environment}")
            
            metadata = self.model_registry[model_id]
            
            # Verify model is validated
            if metadata.status != ModelStatus.VALIDATED:
                raise ValueError(f"Model {model_id} must be validated before deployment")
            
            if metadata.compliance_status != "Enterprise Compliant":
                raise ValueError(f"Model {model_id} is not enterprise compliant")
            
            # Create deployment package
            deployment_package = self._create_deployment_package(model_id)
            
            # Update model status
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.now()
            metadata.deployment_target = environment
            
            # Record deployment
            deployment_record = {
                'model_id': model_id,
                'deployment_time': datetime.now().isoformat(),
                'environment': environment,
                'deployment_status': 'SUCCESS',
                'notes': deployment_notes,
                'package_path': deployment_package
            }
            
            # Update database
            self._update_model_in_database(metadata)
            self._record_deployment_history(deployment_record)
            
            self.logger.info(f"✅ Model deployed successfully: {model_id}")
            self.logger.info(f"   🎯 Environment: {environment}")
            self.logger.info(f"   📦 Package: {deployment_package}")
            
            return deployment_record
            
        except Exception as e:
            self.logger.error(f"❌ Model deployment failed: {e}")
            raise
    
    def load_model(self, model_id: str) -> Any:
        """
        Load model from registry
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model object
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            model_path = Path(metadata.file_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Verify file integrity
            current_hash = self._calculate_file_hash(model_path)
            if current_hash != metadata.file_hash:
                self.logger.warning(f"⚠️ Model file hash mismatch for {model_id}")
            
            # Load model
            model = joblib.load(model_path)
            
            self.logger.info(f"✅ Model loaded: {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_id}: {e}")
            raise
    
    def list_models(self, 
                   status: Optional[ModelStatus] = None,
                   model_type: Optional[ModelType] = None,
                   min_performance: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        List models with optional filtering
        
        Args:
            status: Filter by model status
            model_type: Filter by model type
            min_performance: Filter by minimum performance
            
        Returns:
            List of model summaries
        """
        try:
            models = []
            
            for model_id, metadata in self.model_registry.items():
                # Apply filters
                if status and metadata.status != status:
                    continue
                if model_type and metadata.model_type != model_type:
                    continue
                if min_performance and metadata.validation_score < min_performance:
                    continue
                
                model_summary = {
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type.value,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'validation_score': metadata.validation_score,
                    'created_at': metadata.created_at.isoformat(),
                    'compliance_status': metadata.compliance_status,
                    'business_purpose': metadata.business_purpose,
                    'file_size_mb': round(metadata.file_size / (1024*1024), 2)
                }
                models.append(model_summary)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list models: {e}")
            raise
    
    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get all registered models in the model registry
        
        Returns:
            List of registered model information
        """
        try:
            models = []
            
            for model_id, metadata in self.model_registry.items():
                model_info = {
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type.value,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'validation_score': metadata.validation_score,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'compliance_status': metadata.compliance_status,
                    'business_purpose': metadata.business_purpose,
                    'file_path': metadata.file_path,
                    'file_size_mb': round(metadata.file_size / (1024*1024), 2) if metadata.file_size else 0,
                    'performance_metrics': metadata.performance_metrics,
                    'dependencies': metadata.dependencies,
                    'framework_versions': metadata.framework_versions,
                    'tags': metadata.tags,
                    'model_hash': metadata.model_hash
                }
                models.append(model_info)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            
            self.logger.info(f"📋 Retrieved {len(models)} registered models")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get registered models: {e}")
            return []
    
    def archive_model(self, model_id: str, reason: str = "") -> bool:
        """
        Archive model (move to archive, update status)
        
        Args:
            model_id: Model identifier
            reason: Reason for archiving
            
        Returns:
            Success status
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Move model file to archive
            current_path = Path(metadata.file_path)
            archive_path = self.archive_dir / current_path.name
            
            shutil.move(str(current_path), str(archive_path))
            
            # Update metadata
            metadata.status = ModelStatus.ARCHIVED
            metadata.file_path = str(archive_path)
            metadata.updated_at = datetime.now()
            
            # Update database
            self._update_model_in_database(metadata)
            
            self.logger.info(f"📦 Model archived: {model_id}")
            if reason:
                self.logger.info(f"   📝 Reason: {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to archive model {model_id}: {e}")
            return False
    
    def get_model_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get performance history for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of performance records
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, metric_name, metric_value, dataset_name
                FROM performance_history
                WHERE model_id = ?
                ORDER BY timestamp DESC
            ''', (model_id,))
            
            records = cursor.fetchall()
            conn.close()
            
            performance_history = []
            for record in records:
                performance_history.append({
                    'timestamp': record[0],
                    'metric_name': record[1],
                    'metric_value': record[2],
                    'dataset_name': record[3]
                })
            
            return performance_history
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance history: {e}")
            return []
    
    def generate_model_report(self, model_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive model report
        
        Args:
            model_id: Model identifier
            
        Returns:
            Comprehensive model report
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            performance_history = self.get_model_performance_history(model_id)
            
            # Get deployment history
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT deployment_time, environment, deployment_status, notes
                FROM deployment_history
                WHERE model_id = ?
                ORDER BY deployment_time DESC
            ''', (model_id,))
            deployment_history = [{'deployment_time': r[0], 'environment': r[1], 
                                 'status': r[2], 'notes': r[3]} for r in cursor.fetchall()]
            conn.close()
            
            report = {
                'model_info': metadata.to_dict(),
                'performance_history': performance_history,
                'deployment_history': deployment_history,
                'file_integrity': {
                    'file_exists': Path(metadata.file_path).exists(),
                    'file_size': metadata.file_size,
                    'file_hash': metadata.file_hash
                },
                'compliance_status': {
                    'status': metadata.compliance_status,
                    'risk_level': metadata.risk_level,
                    'business_purpose': metadata.business_purpose
                },
                'report_generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate model report: {e}")
            raise
    
    def cleanup_old_models(self, days_threshold: int = 30, keep_deployed: bool = True) -> Dict[str, Any]:
        """
        Cleanup old models beyond threshold
        
        Args:
            days_threshold: Days threshold for cleanup
            keep_deployed: Whether to keep deployed models
            
        Returns:
            Cleanup summary
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            cleanup_summary = {
                'cleaned_models': [],
                'kept_models': [],
                'total_space_freed': 0,
                'cleanup_date': datetime.now().isoformat()
            }
            
            for model_id, metadata in self.model_registry.items():
                # Skip if too recent
                if metadata.created_at > cutoff_date:
                    cleanup_summary['kept_models'].append({
                        'model_id': model_id,
                        'reason': 'Too recent'
                    })
                    continue
                
                # Skip if deployed and keep_deployed is True
                if keep_deployed and metadata.status == ModelStatus.DEPLOYED:
                    cleanup_summary['kept_models'].append({
                        'model_id': model_id,
                        'reason': 'Currently deployed'
                    })
                    continue
                
                # Archive the model
                file_size = metadata.file_size
                if self.archive_model(model_id, f"Automated cleanup - older than {days_threshold} days"):
                    cleanup_summary['cleaned_models'].append({
                        'model_id': model_id,
                        'file_size': file_size,
                        'age_days': (datetime.now() - metadata.created_at).days
                    })
                    cleanup_summary['total_space_freed'] += file_size
            
            self.logger.info(f"🧹 Cleanup completed: {len(cleanup_summary['cleaned_models'])} models archived")
            self.logger.info(f"   💾 Space freed: {cleanup_summary['total_space_freed']:,} bytes")
            
            return cleanup_summary
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            raise
    
    # === Private Helper Methods ===
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_current_dependencies(self) -> List[str]:
        """Get current Python dependencies"""
        try:
            import pkg_resources
            installed_packages = [str(d) for d in pkg_resources.working_set]
            return installed_packages[:20]  # Limit to prevent bloat
        except Exception:
            return ["dependencies_unavailable"]
    
    def _get_framework_versions(self) -> str:
        """Get ML framework versions"""
        versions = {}
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            versions['tensorflow'] = tf.__version__
        except ImportError:
            pass
        
        try:
            import torch
            versions['torch'] = torch.__version__
        except ImportError:
            pass
        
        return json.dumps(versions)
    
    def _create_model_backup(self, model_id: str) -> str:
        """Create model backup"""
        try:
            metadata = self.model_registry[model_id]
            source_path = Path(metadata.file_path)
            backup_path = self.backups_dir / f"{model_id}_backup.joblib"
            
            shutil.copy2(source_path, backup_path)
            
            # Update metadata
            metadata.backup_path = str(backup_path)
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create backup for {model_id}: {e}")
            return ""
    
    def _create_deployment_package(self, model_id: str) -> str:
        """Create deployment package"""
        try:
            metadata = self.model_registry[model_id]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"{model_id}_deployment_{timestamp}"
            package_path = self.deployment_dir / package_name
            package_path.mkdir(exist_ok=True)
            
            # Copy model file
            source_path = Path(metadata.file_path)
            shutil.copy2(source_path, package_path / "model.joblib")
            
            # Copy metadata
            metadata_source = self.metadata_dir / f"{model_id}_metadata.json"
            shutil.copy2(metadata_source, package_path / "metadata.json")
            
            # Create deployment manifest
            manifest = {
                'model_id': model_id,
                'deployment_time': datetime.now().isoformat(),
                'model_file': 'model.joblib',
                'metadata_file': 'metadata.json',
                'python_version': sys.version,
                'instructions': 'Load using joblib.load("model.joblib")'
            }
            
            with open(package_path / "deployment_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return str(package_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create deployment package: {e}")
            raise
    
    def _save_model_to_database(self, metadata: ModelMetadata):
        """Save model metadata to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_registry (
                    model_id, model_name, model_type, version, status,
                    created_at, updated_at, file_path, file_size, file_hash,
                    performance_metrics, validation_score, training_duration,
                    training_config, feature_count, data_samples,
                    business_purpose, deployment_target, risk_level,
                    compliance_status, dependencies, python_version,
                    framework_version, backup_path, recovery_tested
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id, metadata.model_name, metadata.model_type.value,
                metadata.version, metadata.status.value, metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(), metadata.file_path, metadata.file_size,
                metadata.file_hash, json.dumps(metadata.performance_metrics),
                metadata.validation_score, metadata.training_duration,
                json.dumps(metadata.training_config), metadata.feature_count,
                metadata.data_samples, metadata.business_purpose, metadata.deployment_target,
                metadata.risk_level, metadata.compliance_status,
                json.dumps(metadata.dependencies), metadata.python_version,
                metadata.framework_version, metadata.backup_path,
                1 if metadata.recovery_tested else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model to database: {e}")
            raise
    
    def _update_model_in_database(self, metadata: ModelMetadata):
        """Update model metadata in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE model_registry SET
                    status = ?, updated_at = ?, compliance_status = ?,
                    backup_path = ?, recovery_tested = ?
                WHERE model_id = ?
            ''', (
                metadata.status.value, metadata.updated_at.isoformat(),
                metadata.compliance_status, metadata.backup_path,
                1 if metadata.recovery_tested else 0, metadata.model_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update model in database: {e}")
            raise
    
    def _record_performance_history(self, model_id: str, metrics: Dict[str, float], dataset_name: str):
        """Record performance metrics in history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO performance_history (model_id, timestamp, metric_name, metric_value, dataset_name)
                    VALUES (?, ?, ?, ?, ?)
                ''', (model_id, timestamp, metric_name, metric_value, dataset_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record performance history: {e}")
    
    def _record_deployment_history(self, deployment_record: Dict[str, Any]):
        """Record deployment in history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO deployment_history (model_id, deployment_time, environment, deployment_status, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                deployment_record['model_id'], deployment_record['deployment_time'],
                deployment_record['environment'], deployment_record['deployment_status'],
                deployment_record['notes']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record deployment history: {e}")
    
    def _load_model_registry(self):
        """Load existing models from database"""
        try:
            if not self.db_path.exists():
                return
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM model_registry')
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            for row in rows:
                row_dict = dict(zip(column_names, row))
                
                # Convert back to proper types
                metadata = ModelMetadata(
                    model_id=row_dict['model_id'],
                    model_name=row_dict['model_name'],
                    model_type=ModelType(row_dict['model_type']),
                    version=row_dict['version'],
                    status=ModelStatus(row_dict['status']),
                    created_at=datetime.fromisoformat(row_dict['created_at']),
                    updated_at=datetime.fromisoformat(row_dict['updated_at']),
                    file_path=row_dict['file_path'],
                    file_size=row_dict['file_size'],
                    file_hash=row_dict['file_hash'],
                    performance_metrics=json.loads(row_dict['performance_metrics'] or '{}'),
                    validation_score=row_dict['validation_score'] or 0.0,
                    training_duration=row_dict['training_duration'] or 0.0,
                    training_config=json.loads(row_dict['training_config'] or '{}'),
                    feature_count=row_dict['feature_count'] or 0,
                    data_samples=row_dict['data_samples'] or 0,
                    business_purpose=row_dict['business_purpose'] or '',
                    deployment_target=row_dict['deployment_target'] or '',
                    risk_level=row_dict['risk_level'] or 'Medium',
                    compliance_status=row_dict['compliance_status'] or 'Pending',
                    dependencies=json.loads(row_dict['dependencies'] or '[]'),
                    python_version=row_dict['python_version'] or '',
                    framework_version=row_dict['framework_version'] or '',
                    backup_path=row_dict['backup_path'],
                    recovery_tested=bool(row_dict['recovery_tested'])
                )
                
                self.model_registry[metadata.model_id] = metadata
            
            self.logger.info(f"📚 Loaded {len(self.model_registry)} models from registry")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load model registry: {e}")
    
    def generate_enterprise_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive enterprise model management report
        
        Returns:
            Enterprise report dictionary
        """
        if ENTERPRISE_LOGGER_AVAILABLE:
            with menu1_step_context(Menu1Step.REPORT_GENERATION):
                return self._generate_enterprise_report_internal()
        else:
            return self._generate_enterprise_report_internal()
    
    def _generate_enterprise_report_internal(self):
        """Internal enterprise report generation logic"""
        try:
            self.logger.info("📊 Generating Enterprise Model Report...")
            
            # Model statistics
            total_models = len(self.model_registry)
            status_counts = {}
            type_counts = {}
            compliance_counts = {}
            
            for metadata in self.model_registry.values():
                # Status statistics
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Type statistics
                model_type = metadata.model_type.value
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
                
                # Compliance statistics
                compliance = metadata.compliance_status
                compliance_counts[compliance] = compliance_counts.get(compliance, 0) + 1
            
            # Performance statistics
            performance_stats = self._calculate_performance_statistics()
            
            # Risk analysis
            risk_analysis = self._calculate_risk_analysis()
            
            # Compliance summary
            compliance_summary = self._generate_compliance_summary()
            
            # Deployment summary
            deployment_summary = self._generate_deployment_summary()
            
            # Storage analysis
            storage_analysis = self._calculate_storage_analysis()
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'generated_by': 'Enterprise Model Manager',
                    'report_version': '1.0.0',
                    'total_models': total_models
                },
                'model_statistics': {
                    'total_models': total_models,
                    'status_distribution': status_counts,
                    'type_distribution': type_counts,
                    'compliance_distribution': compliance_counts
                },
                'performance_statistics': performance_stats,
                'risk_analysis': risk_analysis,
                'compliance_summary': compliance_summary,
                'deployment_summary': deployment_summary,
                'storage_analysis': storage_analysis,
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            report_file = self.metadata_dir / f"enterprise_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Enterprise report generated: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate enterprise report: {e}")
            raise
    
    def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics across all models"""
        try:
            validation_scores = []
            training_durations = []
            
            for metadata in self.model_registry.values():
                if metadata.validation_score is not None:
                    validation_scores.append(metadata.validation_score)
                if metadata.training_duration is not None:
                    training_durations.append(metadata.training_duration)
            
            if validation_scores:
                avg_performance = sum(validation_scores) / len(validation_scores)
                max_performance = max(validation_scores)
                min_performance = min(validation_scores)
            else:
                avg_performance = max_performance = min_performance = 0.0
            
            if training_durations:
                avg_training_time = sum(training_durations) / len(training_durations)
                max_training_time = max(training_durations)
                min_training_time = min(training_durations)
            else:
                avg_training_time = max_training_time = min_training_time = 0.0
            
            return {
                'average_performance': round(avg_performance, 4),
                'max_performance': round(max_performance, 4),
                'min_performance': round(min_performance, 4),
                'average_training_time': round(avg_training_time, 2),
                'max_training_time': round(max_training_time, 2),
                'min_training_time': round(min_training_time, 2),
                'total_models_with_performance': len(validation_scores)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance statistics: {e}")
            return {}
    
    def _calculate_risk_analysis(self) -> Dict[str, Any]:
        """Calculate risk analysis across all models"""
        try:
            risk_levels = {}
            high_risk_models = []
            compliance_issues = []
            
            for metadata in self.model_registry.values():
                # Risk level distribution
                risk = metadata.risk_level
                risk_levels[risk] = risk_levels.get(risk, 0) + 1
                
                # High risk models
                if risk == "High":
                    high_risk_models.append({
                        'model_id': metadata.model_id,
                        'model_name': metadata.model_name,
                        'reason': 'High risk classification'
                    })
                
                # Compliance issues
                if metadata.compliance_status not in ["Enterprise Compliant", "Pending"]:
                    compliance_issues.append({
                        'model_id': metadata.model_id,
                        'model_name': metadata.model_name,
                        'issue': metadata.compliance_status
                    })
            
            return {
                'risk_distribution': risk_levels,
                'high_risk_models': high_risk_models,
                'compliance_issues': compliance_issues,
                'total_high_risk': len(high_risk_models),
                'total_compliance_issues': len(compliance_issues)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate risk analysis: {e}")
            return {}
    
    def _generate_compliance_summary(self) -> Dict[str, Any]:
        """Generate compliance summary"""
        try:
            compliant_models = 0
            pending_models = 0
            non_compliant_models = 0
            
            for metadata in self.model_registry.values():
                if metadata.compliance_status == "Enterprise Compliant":
                    compliant_models += 1
                elif metadata.compliance_status == "Pending":
                    pending_models += 1
                else:
                    non_compliant_models += 1
            
            total_models = len(self.model_registry)
            compliance_rate = (compliant_models / total_models * 100) if total_models > 0 else 0
            
            return {
                'compliant_models': compliant_models,
                'pending_models': pending_models,
                'non_compliant_models': non_compliant_models,
                'compliance_rate': round(compliance_rate, 2),
                'total_models': total_models
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate compliance summary: {e}")
            return {}
    
    def _generate_deployment_summary(self) -> Dict[str, Any]:
        """Generate deployment summary"""
        try:
            deployed_models = 0
            validated_models = 0
            production_ready = 0
            
            for metadata in self.model_registry.values():
                if metadata.status == ModelStatus.DEPLOYED:
                    deployed_models += 1
                elif metadata.status == ModelStatus.VALIDATED:
                    validated_models += 1
                
                if (metadata.status == ModelStatus.VALIDATED and 
                    metadata.compliance_status == "Enterprise Compliant"):
                    production_ready += 1
            
            return {
                'deployed_models': deployed_models,
                'validated_models': validated_models,
                'production_ready': production_ready,
                'deployment_rate': round((deployed_models / len(self.model_registry) * 100), 2) if self.model_registry else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate deployment summary: {e}")
            return {}
    
    def _calculate_storage_analysis(self) -> Dict[str, Any]:
        """Calculate storage analysis"""
        try:
            total_size = 0
            file_count = 0
            
            for metadata in self.model_registry.values():
                if metadata.file_size:
                    total_size += metadata.file_size
                    file_count += 1
            
            avg_size = total_size / file_count if file_count > 0 else 0
            
            return {
                'total_storage_bytes': total_size,
                'total_storage_mb': round(total_size / (1024*1024), 2),
                'total_storage_gb': round(total_size / (1024*1024*1024), 2),
                'average_model_size_mb': round(avg_size / (1024*1024), 2),
                'total_model_files': file_count
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate storage analysis: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for model management"""
        recommendations = []
        
        try:
            # Check for outdated models
            outdated_threshold = datetime.now() - timedelta(days=30)
            outdated_models = [m for m in self.model_registry.values() 
                             if m.updated_at < outdated_threshold]
            
            if outdated_models:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'medium',
                    'title': 'Outdated Models Detected',
                    'description': f'{len(outdated_models)} models haven\'t been updated in 30+ days',
                    'action': 'Review and update or archive outdated models'
                })
            
            # Check for low-performance models
            low_performance_models = [m for m in self.model_registry.values() 
                                    if m.validation_score and m.validation_score < 0.7]
            
            if low_performance_models:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'title': 'Low Performance Models',
                    'description': f'{len(low_performance_models)} models have validation scores below 0.7',
                    'action': 'Retrain or replace low-performing models'
                })
            
            # Check for non-compliant models
            non_compliant_models = [m for m in self.model_registry.values() 
                                  if m.compliance_status not in ["Enterprise Compliant", "Pending"]]
            
            if non_compliant_models:
                recommendations.append({
                    'type': 'compliance',
                    'priority': 'high',
                    'title': 'Compliance Issues',
                    'description': f'{len(non_compliant_models)} models have compliance issues',
                    'action': 'Address compliance issues before deployment'
                })
            
            # Check for backup recommendations
            no_backup_models = [m for m in self.model_registry.values() 
                              if not m.backup_path]
            
            if no_backup_models:
                recommendations.append({
                    'type': 'backup',
                    'priority': 'medium',
                    'title': 'Missing Backups',
                    'description': f'{len(no_backup_models)} models don\'t have backups',
                    'action': 'Create backups for all production models'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate recommendations: {e}")
            return []
    
    def get_model_health_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive health status for a specific model"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.model_registry[model_id]
            
            # File integrity check
            file_exists = Path(metadata.file_path).exists()
            file_integrity = "unknown"
            
            if file_exists:
                current_hash = self._calculate_file_hash(Path(metadata.file_path))
                file_integrity = "valid" if current_hash == metadata.file_hash else "corrupted"
            
            # Performance assessment
            performance_level = "unknown"
            if metadata.validation_score is not None:
                if metadata.validation_score >= 0.9:
                    performance_level = "excellent"
                elif metadata.validation_score >= 0.8:
                    performance_level = "good"
                elif metadata.validation_score >= 0.7:
                    performance_level = "acceptable"
                else:
                    performance_level = "poor"
            
            # Age assessment
            age_days = (datetime.now() - metadata.updated_at).days
            age_status = "fresh" if age_days < 7 else "aging" if age_days < 30 else "old"
            
            # Overall health score
            health_score = self._calculate_health_score(metadata, file_integrity, performance_level, age_status)
            
            return {
                'model_id': model_id,
                'overall_health': health_score,
                'file_integrity': file_integrity,
                'file_exists': file_exists,
                'performance_level': performance_level,
                'age_status': age_status,
                'age_days': age_days,
                'compliance_status': metadata.compliance_status,
                'deployment_status': metadata.status.value,
                'last_updated': metadata.updated_at.isoformat(),
                'recommendations': self._get_model_specific_recommendations(metadata, file_integrity, performance_level, age_status)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get model health status: {e}")
            raise
    
    def _calculate_health_score(self, metadata: ModelMetadata, file_integrity: str, 
                              performance_level: str, age_status: str) -> str:
        """Calculate overall health score for a model"""
        score = 0
        
        # File integrity (30%)
        if file_integrity == "valid":
            score += 30
        elif file_integrity == "unknown":
            score += 15
        
        # Performance (40%)
        perf_scores = {"excellent": 40, "good": 30, "acceptable": 20, "poor": 10, "unknown": 0}
        score += perf_scores.get(performance_level, 0)
        
        # Age (20%)
        age_scores = {"fresh": 20, "aging": 15, "old": 5}
        score += age_scores.get(age_status, 0)
        
        # Compliance (10%)
        if metadata.compliance_status == "Enterprise Compliant":
            score += 10
        elif metadata.compliance_status == "Pending":
            score += 5
        
        # Determine health level
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _get_model_specific_recommendations(self, metadata: ModelMetadata, file_integrity: str, 
                                         performance_level: str, age_status: str) -> List[str]:
        """Get specific recommendations for a model"""
        recommendations = []
        
        if file_integrity == "corrupted":
            recommendations.append("File integrity compromised - restore from backup")
        elif file_integrity == "unknown":
            recommendations.append("Verify file integrity")
        
        if performance_level == "poor":
            recommendations.append("Performance below acceptable - consider retraining")
        elif performance_level == "unknown":
            recommendations.append("Performance not evaluated - run validation")
        
        if age_status == "old":
            recommendations.append("Model is outdated - consider retraining with fresh data")
        
        if metadata.compliance_status not in ["Enterprise Compliant", "Pending"]:
            recommendations.append("Address compliance issues before deployment")
        
        if not metadata.backup_path:
            recommendations.append("Create backup for disaster recovery")
        
        return recommendations
    
    def bulk_validate_models(self, validation_data: pd.DataFrame, 
                           validation_target: pd.Series) -> Dict[str, Any]:
        """Validate multiple models in bulk"""
        if ENTERPRISE_LOGGER_AVAILABLE:
            with menu1_step_context(Menu1Step.BATCH_PROCESSING):
                return self._bulk_validate_models_internal(validation_data, validation_target)
        else:
            return self._bulk_validate_models_internal(validation_data, validation_target)
    
    def _bulk_validate_models_internal(self, validation_data, validation_target):
        """Internal bulk validation logic"""
        try:
            self.logger.info(f"🔍 Starting bulk validation of {len(self.model_registry)} models...")
            
            validation_results = {}
            successful_validations = 0
            failed_validations = 0
            
            for model_id in self.model_registry:
                try:
                    result = self._validate_model_internal(
                        model_id, validation_data, validation_target, 0.7
                    )
                    validation_results[model_id] = result
                    
                    if result['passed']:
                        successful_validations += 1
                    else:
                        failed_validations += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Validation failed for {model_id}: {e}")
                    validation_results[model_id] = {
                        'passed': False,
                        'error': str(e)
                    }
                    failed_validations += 1
            
            summary = {
                'total_models': len(self.model_registry),
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'success_rate': round((successful_validations / len(self.model_registry) * 100), 2) if self.model_registry else 0,
                'individual_results': validation_results
            }
            
            self.logger.info(f"✅ Bulk validation complete: {successful_validations}/{len(self.model_registry)} passed")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Bulk validation failed: {e}")
            raise
    
    def export_model_registry(self, export_path: str = None) -> str:
        """Export model registry to JSON file"""
        try:
            if export_path is None:
                export_path = self.metadata_dir / f"model_registry_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                'export_metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_models': len(self.model_registry),
                    'export_version': '1.0.0'
                },
                'models': {}
            }
            
            for model_id, metadata in self.model_registry.items():
                export_data['models'][model_id] = metadata.to_dict()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📤 Model registry exported to: {export_path}")
            
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export model registry: {e}")
            raise

# Global model manager instance
_model_manager = None

def get_enterprise_model_manager(config: Dict[str, Any] = None, logger: logging.Logger = None) -> EnterpriseModelManager:
    """
    Get singleton instance of Enterprise Model Manager
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        EnterpriseModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = EnterpriseModelManager(config, logger)
    return _model_manager

# Export classes and functions
__all__ = [
    'EnterpriseModelManager',
    'ModelMetadata', 
    'ModelStatus',
    'ModelType',
    'get_enterprise_model_manager'
]
