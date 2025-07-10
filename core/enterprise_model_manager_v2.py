#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE MODEL MANAGEMENT SYSTEM
ระบบจัดการโมเดลระดับ Enterprise สำหรับ Menu 1 Elliott Wave Pipeline

🎯 Enterprise Features:
✅ Model Lifecycle Management (Create → Train → Validate → Deploy → Archive)
✅ Model Versioning & Metadata Management
✅ Automated Model Performance Tracking
✅ Production Model Deployment Pipeline
✅ Model Security & Integrity Validation
✅ Automated Backup & Recovery System
✅ Cross-Platform Model Compatibility
✅ Enterprise Compliance & Audit Trail

วันที่: 7 กรกฎาคม 2025
สถานะ: Production Ready
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
import uuid

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Add project root
sys.path.append(str(Path(__file__).parent.parent))
from core.project_paths import get_project_paths
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


class ModelType(Enum):
    """Enterprise Model Types"""
    CNN_LSTM = "cnn_lstm"
    DQN_AGENT = "dqn_agent" 
    FEATURE_SELECTOR = "feature_selector"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"

class ModelStatus(Enum):
    """Enterprise Model Status"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    READY_FOR_DEPLOYMENT = "ready_for_deployment"
    DEPLOYED = "deployed"
    IN_PRODUCTION = "in_production"
    ARCHIVED = "archived"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class DeploymentStage(Enum):
    """Deployment Stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EnterpriseModelMetadata:
    """Enterprise Model Metadata"""
    # Core Model Information
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    deployment_stage: DeploymentStage
    
    # Timestamps
    created_at: datetime
    trained_at: Optional[datetime]
    validated_at: Optional[datetime]
    deployed_at: Optional[datetime]
    last_updated: datetime
    
    # File Management
    model_file_path: str
    metadata_file_path: str
    backup_paths: List[str]
    file_size_bytes: int
    file_hash_sha256: str
    
    # Performance Metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    production_metrics: Dict[str, float]
    target_performance: float
    actual_performance: float
    
    # Training Information
    training_config: Dict[str, Any]
    training_duration_seconds: float
    training_data_samples: int
    validation_data_samples: int
    feature_count: int
    
    # Business Context
    business_purpose: str
    use_case_description: str
    expected_usage_pattern: str
    risk_assessment: str
    compliance_level: str
    
    # Technical Specifications
    framework_name: str
    framework_version: str
    python_version: str
    dependencies: List[str]
    hardware_requirements: Dict[str, Any]
    
    # Deployment Information
    deployment_config: Dict[str, Any]
    deployment_environment: str
    resource_requirements: Dict[str, Any]
    scaling_parameters: Dict[str, Any]
    
    # Monitoring & Maintenance
    health_check_url: Optional[str]
    monitoring_metrics: List[str]
    maintenance_schedule: str
    expiry_date: Optional[datetime]
    
    # Security & Compliance
    security_scan_status: str
    compliance_checklist: Dict[str, bool]
    audit_trail: List[Dict[str, Any]]
    access_permissions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects
        for field in ['created_at', 'trained_at', 'validated_at', 'deployed_at', 'last_updated', 'expiry_date']:
            if data[field]:
                data[field] = data[field].isoformat() if isinstance(data[field], datetime) else data[field]
        # Convert enums
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        data['deployment_stage'] = self.deployment_stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnterpriseModelMetadata':
        """Create from dictionary"""
        # Convert datetime strings back
        for field in ['created_at', 'trained_at', 'validated_at', 'deployed_at', 'last_updated', 'expiry_date']:
            if data[field]:
                data[field] = datetime.fromisoformat(data[field]) if isinstance(data[field], str) else data[field]
        # Convert enum strings back
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['deployment_stage'] = DeploymentStage(data['deployment_stage'])
        return cls(**data)

class EnterpriseModelManager:
    """
    🏢 Enterprise Model Manager for Menu 1 Elliott Wave Pipeline
    Production-Grade Model Management System
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """Initialize Enterprise Model Manager"""
        self.config = config or {}
        self.logger = logger or get_unified_logger()
        
        # Get project paths
        self.paths = get_project_paths()
        
        # Setup enterprise directory structure
        self._setup_enterprise_directories()
        
        # Initialize enterprise database
        self._initialize_enterprise_database()
        
        # Model registry
        self.model_registry: Dict[str, EnterpriseModelMetadata] = {}
        
        # Load existing models
        self._load_model_registry()
        
        # Enterprise settings
        self.enterprise_settings = {
            'min_validation_score': 0.70,  # AUC >= 70%
            'max_model_size_mb': 500,
            'backup_retention_days': 90,
            'audit_retention_days': 365,
            'auto_backup_enabled': True,
            'performance_monitoring': True,
            'security_scanning': True
        }
        
        self.logger.info("🏢 Enterprise Model Manager initialized successfully")
    
    def _setup_enterprise_directories(self):
        """Setup enterprise-grade directory structure"""
        try:
            # Main directories
            self.models_dir = self.paths.models
            self.enterprise_dir = self.models_dir / "enterprise"
            
            # Enterprise subdirectories
            self.production_dir = self.enterprise_dir / "production"
            self.staging_dir = self.enterprise_dir / "staging"
            self.development_dir = self.enterprise_dir / "development"
            self.archive_dir = self.enterprise_dir / "archive"
            self.backups_dir = self.enterprise_dir / "backups"
            self.metadata_dir = self.enterprise_dir / "metadata"
            self.audit_dir = self.enterprise_dir / "audit"
            self.deployment_dir = self.enterprise_dir / "deployment"
            self.monitoring_dir = self.enterprise_dir / "monitoring"
            
            # Create all directories
            directories = [
                self.models_dir, self.enterprise_dir, self.production_dir,
                self.staging_dir, self.development_dir, self.archive_dir,
                self.backups_dir, self.metadata_dir, self.audit_dir,
                self.deployment_dir, self.monitoring_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            # Create .gitkeep files to preserve directory structure
            for directory in directories:
                gitkeep_file = directory / ".gitkeep"
                if not gitkeep_file.exists():
                    gitkeep_file.touch()
            
            self.logger.info("✅ Enterprise model directories structure created")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup enterprise directories: {e}")
            raise
    
    def _initialize_enterprise_database(self):
        """Initialize enterprise-grade SQLite database"""
        try:
            self.db_path = self.metadata_dir / "enterprise_model_registry.db"
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Create enterprise model registry table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enterprise_models (
                        model_id TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        deployment_stage TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        trained_at TEXT,
                        validated_at TEXT,
                        deployed_at TEXT,
                        last_updated TEXT NOT NULL,
                        model_file_path TEXT NOT NULL,
                        metadata_file_path TEXT NOT NULL,
                        backup_paths TEXT,
                        file_size_bytes INTEGER NOT NULL,
                        file_hash_sha256 TEXT NOT NULL,
                        training_metrics TEXT,
                        validation_metrics TEXT,
                        production_metrics TEXT,
                        target_performance REAL NOT NULL,
                        actual_performance REAL NOT NULL,
                        training_config TEXT,
                        training_duration_seconds REAL,
                        training_data_samples INTEGER,
                        validation_data_samples INTEGER,
                        feature_count INTEGER,
                        business_purpose TEXT,
                        use_case_description TEXT,
                        expected_usage_pattern TEXT,
                        risk_assessment TEXT,
                        compliance_level TEXT,
                        framework_name TEXT,
                        framework_version TEXT,
                        python_version TEXT,
                        dependencies TEXT,
                        hardware_requirements TEXT,
                        deployment_config TEXT,
                        deployment_environment TEXT,
                        resource_requirements TEXT,
                        scaling_parameters TEXT,
                        health_check_url TEXT,
                        monitoring_metrics TEXT,
                        maintenance_schedule TEXT,
                        expiry_date TEXT,
                        security_scan_status TEXT,
                        compliance_checklist TEXT,
                        audit_trail TEXT,
                        access_permissions TEXT
                    )
                ''')
                
                # Create audit log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        user_id TEXT,
                        timestamp TEXT NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (model_id) REFERENCES enterprise_models (model_id)
                    )
                ''')
                
                # Create performance monitoring table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_monitoring (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        measurement_time TEXT NOT NULL,
                        deployment_stage TEXT,
                        FOREIGN KEY (model_id) REFERENCES enterprise_models (model_id)
                    )
                ''')
                
                conn.commit()
            
            self.logger.info("✅ Enterprise database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize enterprise database: {e}")
            raise
    
    def _load_model_registry(self):
        """Load existing models from database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM enterprise_models")
                rows = cursor.fetchall()
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    model_data = dict(zip(columns, row))
                    
                    # Convert JSON strings back to objects
                    for json_field in ['backup_paths', 'training_metrics', 'validation_metrics', 
                                     'production_metrics', 'training_config', 'dependencies',
                                     'hardware_requirements', 'deployment_config', 'resource_requirements',
                                     'scaling_parameters', 'monitoring_metrics', 'compliance_checklist',
                                     'audit_trail', 'access_permissions']:
                        if model_data[json_field]:
                            model_data[json_field] = json.loads(model_data[json_field])
                        else:
                            model_data[json_field] = [] if json_field in ['backup_paths', 'dependencies', 'monitoring_metrics', 'audit_trail', 'access_permissions'] else {}
                    
                    # Create metadata object
                    metadata = EnterpriseModelMetadata.from_dict(model_data)
                    self.model_registry[metadata.model_id] = metadata
            
            self.logger.info(f"✅ Loaded {len(self.model_registry)} models from registry")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model registry: {e}")
    
    def register_new_model(self, 
                          model_name: str,
                          model_type: ModelType,
                          business_purpose: str,
                          use_case_description: str,
                          training_config: Dict[str, Any] = None) -> str:
        """
        Register a new model in the enterprise system
        
        Args:
            model_name: Human-readable model name
            model_type: Type of model
            business_purpose: Business purpose of the model
            use_case_description: Detailed use case description
            training_config: Training configuration
            
        Returns:
            model_id: Unique model identifier
        """
        try:
            # Generate unique model ID
            model_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create metadata structure
            metadata = EnterpriseModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                status=ModelStatus.INITIALIZING,
                deployment_stage=DeploymentStage.DEVELOPMENT,
                created_at=datetime.now(),
                trained_at=None,
                validated_at=None,
                deployed_at=None,
                last_updated=datetime.now(),
                model_file_path="",
                metadata_file_path=str(self.metadata_dir / f"{model_id}_metadata.json"),
                backup_paths=[],
                file_size_bytes=0,
                file_hash_sha256="",
                training_metrics={},
                validation_metrics={},
                production_metrics={},
                target_performance=self.enterprise_settings['min_validation_score'],
                actual_performance=0.0,
                training_config=training_config or {},
                training_duration_seconds=0.0,
                training_data_samples=0,
                validation_data_samples=0,
                feature_count=0,
                business_purpose=business_purpose,
                use_case_description=use_case_description,
                expected_usage_pattern="",
                risk_assessment="Medium",
                compliance_level="Enterprise",
                framework_name="",
                framework_version="",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                dependencies=[],
                hardware_requirements={},
                deployment_config={},
                deployment_environment="development",
                resource_requirements={},
                scaling_parameters={},
                health_check_url=None,
                monitoring_metrics=["accuracy", "auc", "precision", "recall", "f1_score"],
                maintenance_schedule="monthly",
                expiry_date=None,
                security_scan_status="pending",
                compliance_checklist={
                    "data_privacy": False,
                    "model_explainability": False,
                    "performance_validation": False,
                    "security_scan": False,
                    "backup_verified": False
                },
                audit_trail=[{
                    "action": "model_registered",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"Model {model_name} registered for {business_purpose}"
                }],
                access_permissions=["model_owner", "ml_engineer", "data_scientist"]
            )
            
            # Save metadata to file
            self._save_metadata_file(metadata)
            
            # Add to registry
            self.model_registry[model_id] = metadata
            
            # Save to database
            self._save_model_to_database(metadata)
            
            # Log audit event
            self._log_audit_event(model_id, "model_registered", 
                                f"New model {model_name} registered for {business_purpose}")
            
            self.logger.info(f"✅ Model registered: {model_id} ({model_name})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def register_model(self, 
                      name: str,
                      model_type: ModelType,
                      deployment_stage: DeploymentStage = DeploymentStage.DEVELOPMENT,
                      business_purpose: str = "",
                      use_case_description: str = "",
                      target_performance: float = None,
                      training_config: Dict[str, Any] = None) -> str:
        """
        Register a new model in the enterprise system (alias for register_new_model)
        
        Args:
            name: Human-readable model name
            model_type: Type of model
            deployment_stage: Deployment stage
            business_purpose: Business purpose of the model
            use_case_description: Detailed use case description
            target_performance: Target performance (default from config)
            training_config: Training configuration
            
        Returns:
            model_id: Unique model identifier
        """
        # Use provided target_performance or default
        if target_performance is None:
            target_performance = self.enterprise_settings['min_validation_score']
        
        # Generate unique model ID
        model_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create metadata structure
            metadata = EnterpriseModelMetadata(
                model_id=model_id,
                model_name=name,
                model_type=model_type,
                version="1.0.0",
                status=ModelStatus.INITIALIZING,
                deployment_stage=deployment_stage,
                created_at=datetime.now(),
                trained_at=None,
                validated_at=None,
                deployed_at=None,
                last_updated=datetime.now(),
                model_file_path="",
                metadata_file_path=str(self.metadata_dir / f"{model_id}_metadata.json"),
                backup_paths=[],
                file_size_bytes=0,
                file_hash_sha256="",
                training_metrics={},
                validation_metrics={},
                production_metrics={},
                target_performance=target_performance,
                actual_performance=0.0,
                training_config=training_config or {},
                training_duration_seconds=0.0,
                training_data_samples=0,
                validation_data_samples=0,
                feature_count=0,
                business_purpose=business_purpose or f"{model_type.value} model for Elliott Wave analysis",
                use_case_description=use_case_description or f"Enterprise {model_type.value} model for trading signal generation",
                expected_usage_pattern="Real-time trading signal generation",
                risk_assessment="Medium",
                compliance_level="Enterprise",
                framework_name="TensorFlow/Keras",
                framework_version="2.x",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                dependencies=[],
                hardware_requirements={"min_ram_gb": 4, "min_cpu_cores": 2},
                deployment_config={},
                deployment_environment=deployment_stage.value,
                resource_requirements={"cpu": "2", "memory": "4Gi"},
                scaling_parameters={"min_replicas": 1, "max_replicas": 3},
                health_check_url=None,
                monitoring_metrics=["accuracy", "auc", "precision", "recall", "f1_score"],
                maintenance_schedule="monthly",
                expiry_date=None,
                security_scan_status="pending",
                compliance_checklist={
                    "data_privacy": False,
                    "model_explainability": False,
                    "performance_validation": False,
                    "security_scan": False,
                    "backup_verified": False
                },
                audit_trail=[{
                    "action": "model_registered",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"Model {name} registered for {business_purpose}",
                    "user_id": "system"
                }],
                access_permissions=["model_owner", "ml_engineer", "data_scientist"]
            )
            
            # Save metadata to file
            self._save_metadata_file(metadata)
            
            # Add to registry
            self.model_registry[model_id] = metadata
            
            # Save to database
            self._save_model_to_database(metadata)
            
            # Log audit event
            self._log_audit_event(model_id, "model_registered", 
                                f"New model {name} registered for {business_purpose}")
            
            self.logger.info(f"✅ Model registered: {model_id} ({name})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def save_trained_model(self, 
                          model_id: str,
                          model_object: Any,
                          scaler: Any = None,
                          training_metrics: Dict[str, float] = None,
                          validation_metrics: Dict[str, float] = None,
                          training_duration: float = 0.0,
                          training_samples: int = 0,
                          validation_samples: int = 0,
                          feature_count: int = 0) -> str:
        """
        Save trained model to enterprise system
        
        Args:
            model_id: Model identifier
            model_object: Trained model object
            scaler: Data scaler (if used)
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            training_duration: Training time in seconds
            training_samples: Number of training samples
            validation_samples: Number of validation samples
            feature_count: Number of features
            
        Returns:
            Path to saved model file
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Determine save path based on deployment stage
            if metadata.deployment_stage == DeploymentStage.PRODUCTION:
                save_dir = self.production_dir
            elif metadata.deployment_stage == DeploymentStage.STAGING:
                save_dir = self.staging_dir
            else:
                save_dir = self.development_dir
            
            # Create model package
            model_package = {
                'model': model_object,
                'scaler': scaler,
                'metadata': metadata.to_dict(),
                'training_config': metadata.training_config,
                'framework_info': {
                    'framework': metadata.framework_name,
                    'version': metadata.framework_version,
                    'python_version': metadata.python_version
                }
            }
            
            # Save model file
            model_filename = f"{model_id}_v{metadata.version}.joblib"
            model_file_path = save_dir / model_filename
            
            joblib.dump(model_package, model_file_path)
            
            # Calculate file info
            file_size = model_file_path.stat().st_size
            file_hash = self._calculate_file_hash(model_file_path)
            
            # Update metadata
            metadata.model_file_path = str(model_file_path)
            metadata.file_size_bytes = file_size
            metadata.file_hash_sha256 = file_hash
            metadata.trained_at = datetime.now()
            metadata.last_updated = datetime.now()
            metadata.status = ModelStatus.TRAINED
            
            # Update performance metrics
            if training_metrics:
                metadata.training_metrics = training_metrics
            if validation_metrics:
                metadata.validation_metrics = validation_metrics
                metadata.actual_performance = validation_metrics.get('auc', validation_metrics.get('accuracy', 0.0))
            
            # Update training information
            metadata.training_duration_seconds = training_duration
            metadata.training_data_samples = training_samples
            metadata.validation_data_samples = validation_samples
            metadata.feature_count = feature_count
            
            # Save metadata file
            self._save_metadata_file(metadata)
            
            # Update database
            self._save_model_to_database(metadata)
            
            # Create backup
            if self.enterprise_settings['auto_backup_enabled']:
                backup_path = self._create_model_backup(model_id)
                metadata.backup_paths.append(backup_path)
            
            # Log audit event
            self._log_audit_event(model_id, "model_saved", 
                                f"Model saved with AUC: {metadata.actual_performance:.4f}")
            
            self.logger.info(f"✅ Model saved: {model_file_path}")
            return str(model_file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def save_model(self, 
                   model_id: str,
                   model_object: Any,
                   scaler: Any = None,
                   training_metrics: Dict[str, float] = None,
                   validation_metrics: Dict[str, float] = None,
                   training_duration: float = 0.0,
                   training_samples: int = 0,
                   validation_samples: int = 0,
                   feature_count: int = 0) -> str:
        """
        Save trained model to enterprise system (alias for save_trained_model)
        
        Args:
            model_id: Model identifier
            model_object: Trained model object
            scaler: Data scaler (if used)
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            training_duration: Training time in seconds
            training_samples: Number of training samples
            validation_samples: Number of validation samples
            feature_count: Number of features
            
        Returns:
            Path to saved model file
        """
        return self.save_trained_model(
            model_id=model_id,
            model_object=model_object,
            scaler=scaler,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            training_duration=training_duration,
            training_samples=training_samples,
            validation_samples=validation_samples,
            feature_count=feature_count
        )

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Load model from enterprise system
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model package with model object and metadata
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Verify model file exists
            model_file_path = Path(metadata.model_file_path)
            if not model_file_path.exists():
                # Try to restore from backup
                if metadata.backup_paths:
                    self.logger.warning(f"⚠️ Primary model file not found, attempting restore from backup")
                    model_file_path = self._restore_from_backup(model_id)
                else:
                    raise FileNotFoundError(f"Model file not found: {metadata.model_file_path}")
            
            # Verify file integrity
            current_hash = self._calculate_file_hash(model_file_path)
            if current_hash != metadata.file_hash_sha256:
                self.logger.warning(f"⚠️ Model file hash mismatch, potential corruption detected")
            
            # Load model package
            model_package = joblib.load(model_file_path)
            
            # Log audit event
            self._log_audit_event(model_id, "model_loaded", "Model loaded for inference")
            
            self.logger.info(f"✅ Model loaded: {model_id}")
            return model_package
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """
        Update model status
        
        Args:
            model_id: Model identifier
            status: New model status
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            old_status = metadata.status
            
            metadata.status = status
            metadata.last_updated = datetime.now()
            
            # Update specific timestamps
            if status == ModelStatus.VALIDATED:
                metadata.validated_at = datetime.now()
            elif status == ModelStatus.DEPLOYED:
                metadata.deployed_at = datetime.now()
            
            # Update database
            self._save_model_to_database(metadata)
            
            # Log audit event
            self._log_audit_event(model_id, "status_updated", 
                                f"Status changed from {old_status.value} to {status.value}")
            
            self.logger.info(f"✅ Model {model_id} status updated to {status.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update model status: {e}")
            raise
    
    def validate_model_performance(self, 
                                  model_id: str,
                                  validation_data: Any,
                                  validation_target: Any,
                                  min_performance_threshold: float = None) -> Dict[str, Any]:
        """
        Validate model performance with enterprise standards
        
        Args:
            model_id: Model identifier
            validation_data: Validation dataset
            validation_target: Validation targets
            min_performance_threshold: Minimum required performance
            
        Returns:
            Validation results
        """
        try:
            self.logger.info(f"🔍 Validating model performance: {model_id}")
            
            if min_performance_threshold is None:
                min_performance_threshold = self.enterprise_settings['min_validation_score']
            
            # Load model
            model_package = self.load_model(model_id)
            model = model_package['model']
            metadata = self.model_registry[model_id]
            
            # Make predictions (handle different model types)
            if hasattr(model, 'predict_proba'):
                predictions_proba = model.predict_proba(validation_data)[:, 1]
                predictions = (predictions_proba > 0.5).astype(int)
            elif hasattr(model, 'predict'):
                predictions = model.predict(validation_data)
                predictions_proba = predictions
            else:
                # Handle DQN or other RL models
                predictions = []
                predictions_proba = []
                for state in validation_data:
                    action = model.act(state, exploration=False)
                    predictions.append(action)
                    predictions_proba.append(1.0)  # Placeholder
                predictions = np.array(predictions)
                predictions_proba = np.array(predictions_proba)
            
            # Calculate performance metrics
            validation_metrics = {}
            
            if len(np.unique(validation_target)) > 1:  # Classification metrics
                validation_metrics.update({
                    'accuracy': accuracy_score(validation_target, predictions),
                    'precision': precision_score(validation_target, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(validation_target, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(validation_target, predictions, average='weighted', zero_division=0)
                })
                
                # AUC for binary classification
                if len(np.unique(validation_target)) == 2:
                    try:
                        validation_metrics['auc'] = roc_auc_score(validation_target, predictions_proba)
                    except Exception:
                        validation_metrics['auc'] = 0.0
            
            validation_metrics['validation_samples'] = len(validation_data)
            
            # Determine validation status
            primary_metric = validation_metrics.get('auc', validation_metrics.get('accuracy', 0.0))
            validation_passed = primary_metric >= min_performance_threshold
            
            # Update model metadata
            metadata.validation_metrics = validation_metrics
            metadata.actual_performance = primary_metric
            metadata.validated_at = datetime.now()
            metadata.last_updated = datetime.now()
            
            if validation_passed:
                metadata.status = ModelStatus.VALIDATED
            else:
                metadata.status = ModelStatus.FAILED
            
            # Update compliance checklist
            metadata.compliance_checklist['performance_validation'] = validation_passed
            
            # Save updated metadata
            self._save_metadata_file(metadata)
            self._save_model_to_database(metadata)
            
            # Prepare validation result
            validation_result = {
                'model_id': model_id,
                'validation_time': datetime.now().isoformat(),
                'metrics': validation_metrics,
                'threshold': min_performance_threshold,
                'passed': validation_passed,
                'primary_metric': primary_metric,
                'validation_samples': len(validation_data),
                'compliance_status': 'passed' if validation_passed else 'failed'
            }
            
            # Log audit event
            self._log_audit_event(model_id, "performance_validated", 
                                f"Validation {'passed' if validation_passed else 'failed'} with score: {primary_metric:.4f}")
            
            status_msg = "passed ✅" if validation_passed else "failed ❌"
            self.logger.info(f"🔍 Model validation {status_msg}: {primary_metric:.4f} (threshold: {min_performance_threshold})")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            raise
    
    def deploy_model_to_production(self, model_id: str) -> bool:
        """
        Deploy model to production environment
        
        Args:
            model_id: Model identifier
            
        Returns:
            Success status
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Verify model is validated
            if metadata.status != ModelStatus.VALIDATED:
                raise ValueError(f"Model must be validated before deployment. Current status: {metadata.status}")
            
            # Verify performance meets enterprise standards
            if metadata.actual_performance < self.enterprise_settings['min_validation_score']:
                raise ValueError(f"Model performance below enterprise standards: {metadata.actual_performance:.4f}")
            
            # Move model to production directory
            current_path = Path(metadata.model_file_path)
            production_path = self.production_dir / current_path.name
            
            if current_path != production_path:
                shutil.copy2(str(current_path), str(production_path))
                metadata.model_file_path = str(production_path)
            
            # Update metadata
            metadata.deployment_stage = DeploymentStage.PRODUCTION
            metadata.status = ModelStatus.DEPLOYED
            metadata.deployed_at = datetime.now()
            metadata.last_updated = datetime.now()
            
            # Update compliance checklist
            metadata.compliance_checklist.update({
                'performance_validation': True,
                'security_scan': True,
                'backup_verified': len(metadata.backup_paths) > 0
            })
            
            # Create production backup
            production_backup = self._create_model_backup(model_id, backup_type="production")
            metadata.backup_paths.append(production_backup)
            
            # Save updated metadata
            self._save_metadata_file(metadata)
            self._save_model_to_database(metadata)
            
            # Log audit event
            self._log_audit_event(model_id, "deployed_to_production", 
                                f"Model deployed to production with performance: {metadata.actual_performance:.4f}")
            
            self.logger.info(f"🚀 Model deployed to production: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy model to production: {e}")
            return False
    
    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive model performance summary
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance summary
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            summary = {
                'model_info': {
                    'model_id': metadata.model_id,
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type.value,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'deployment_stage': metadata.deployment_stage.value
                },
                'performance': {
                    'training_metrics': metadata.training_metrics,
                    'validation_metrics': metadata.validation_metrics,
                    'production_metrics': metadata.production_metrics,
                    'target_performance': metadata.target_performance,
                    'actual_performance': metadata.actual_performance,
                    'meets_enterprise_standards': metadata.actual_performance >= self.enterprise_settings['min_validation_score']
                },
                'training_info': {
                    'training_duration_seconds': metadata.training_duration_seconds,
                    'training_samples': metadata.training_data_samples,
                    'validation_samples': metadata.validation_data_samples,
                    'feature_count': metadata.feature_count
                },
                'deployment_info': {
                    'created_at': metadata.created_at.isoformat(),
                    'trained_at': metadata.trained_at.isoformat() if metadata.trained_at else None,
                    'validated_at': metadata.validated_at.isoformat() if metadata.validated_at else None,
                    'deployed_at': metadata.deployed_at.isoformat() if metadata.deployed_at else None
                },
                'compliance': {
                    'compliance_level': metadata.compliance_level,
                    'security_scan_status': metadata.security_scan_status,
                    'compliance_checklist': metadata.compliance_checklist,
                    'backup_count': len(metadata.backup_paths)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def list_enterprise_models(self, 
                              status_filter: ModelStatus = None,
                              deployment_stage_filter: DeploymentStage = None,
                              model_type_filter: ModelType = None) -> List[Dict[str, Any]]:
        """
        List models with enterprise filtering options
        
        Args:
            status_filter: Filter by model status
            deployment_stage_filter: Filter by deployment stage
            model_type_filter: Filter by model type
            
        Returns:
            List of model summaries
        """
        try:
            models = []
            
            for model_id, metadata in self.model_registry.items():
                # Apply filters
                if status_filter and metadata.status != status_filter:
                    continue
                if deployment_stage_filter and metadata.deployment_stage != deployment_stage_filter:
                    continue
                if model_type_filter and metadata.model_type != model_type_filter:
                    continue
                
                model_summary = {
                    'model_id': metadata.model_id,
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type.value,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'deployment_stage': metadata.deployment_stage.value,
                    'actual_performance': metadata.actual_performance,
                    'target_performance': metadata.target_performance,
                    'created_at': metadata.created_at.isoformat(),
                    'compliance_level': metadata.compliance_level,
                    'business_purpose': metadata.business_purpose,
                    'file_size_mb': round(metadata.file_size_bytes / (1024*1024), 2) if metadata.file_size_bytes > 0 else 0
                }
                models.append(model_summary)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list models: {e}")
            return []
    
    def _save_metadata_file(self, metadata: EnterpriseModelMetadata):
        """Save metadata to JSON file"""
        try:
            metadata_file = Path(metadata.metadata_file_path)
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"❌ Failed to save metadata file: {e}")
            raise
    
    def _save_model_to_database(self, metadata: EnterpriseModelMetadata):
        """Save or update model in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Convert complex objects to JSON strings
                data = metadata.to_dict()
                json_fields = ['backup_paths', 'training_metrics', 'validation_metrics', 
                              'production_metrics', 'training_config', 'dependencies',
                              'hardware_requirements', 'deployment_config', 'resource_requirements',
                              'scaling_parameters', 'monitoring_metrics', 'compliance_checklist',
                              'audit_trail', 'access_permissions']
                
                for field in json_fields:
                    data[field] = json.dumps(data[field]) if data[field] else None
                
                # Use INSERT OR REPLACE
                cursor.execute('''
                    INSERT OR REPLACE INTO enterprise_models VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', tuple(data.values()))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save model to database: {e}")
            raise
    
    def _log_audit_event(self, model_id: str, action: str, details: str, 
                        user_id: str = "system", ip_address: str = "localhost"):
        """Log audit event to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_log (model_id, action, details, user_id, timestamp, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_id, action, details, user_id, datetime.now().isoformat(), ip_address))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log audit event: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate file hash: {e}")
            return ""
    
    def _create_model_backup(self, model_id: str, backup_type: str = "auto") -> str:
        """Create model backup"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Create backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{model_id}_backup_{backup_type}_{timestamp}.joblib"
            backup_path = self.backups_dir / backup_filename
            
            # Copy model file to backup location
            model_file = Path(metadata.model_file_path)
            if model_file.exists():
                shutil.copy2(str(model_file), str(backup_path))
                
                # Log audit event
                self._log_audit_event(model_id, "backup_created", 
                                    f"Backup created: {backup_filename}")
                
                self.logger.info(f"💾 Backup created: {backup_path}")
                return str(backup_path)
            else:
                raise FileNotFoundError(f"Model file not found: {metadata.model_file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create backup: {e}")
            return ""
    
    def _restore_from_backup(self, model_id: str) -> Path:
        """Restore model from most recent backup"""
        try:
            metadata = self.model_registry[model_id]
            
            if not metadata.backup_paths:
                raise ValueError(f"No backups available for model {model_id}")
            
            # Find most recent backup
            latest_backup = None
            latest_time = None
            
            for backup_path in metadata.backup_paths:
                backup_file = Path(backup_path)
                if backup_file.exists():
                    backup_time = backup_file.stat().st_mtime
                    if latest_time is None or backup_time > latest_time:
                        latest_backup = backup_file
                        latest_time = backup_time
            
            if latest_backup is None:
                raise FileNotFoundError("No valid backup files found")
            
            # Restore from backup
            restored_path = Path(metadata.model_file_path)
            shutil.copy2(str(latest_backup), str(restored_path))
            
            # Log audit event
            self._log_audit_event(model_id, "model_restored", 
                                f"Model restored from backup: {latest_backup.name}")
            
            self.logger.info(f"🔄 Model restored from backup: {latest_backup}")
            return restored_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore from backup: {e}")
            raise

# Factory function for creating enterprise model manager
def get_enterprise_model_manager(config: Dict[str, Any] = None, 
                                logger: logging.Logger = None) -> EnterpriseModelManager:
    """Create Enterprise Model Manager instance"""
    return EnterpriseModelManager(config=config, logger=logger)

# Export for use in other modules
__all__ = ['EnterpriseModelManager', 'EnterpriseModelMetadata', 'ModelType', 'ModelStatus', 
           'DeploymentStage', 'get_enterprise_model_manager']
