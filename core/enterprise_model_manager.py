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
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    ERROR = "error"
    
class DeploymentStage(Enum):
    """Enterprise Deployment Stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


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
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnterpriseModelMetadata':
        """Create from dictionary"""
        # Convert enums back from strings
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['deployment_stage'] = DeploymentStage(data['deployment_stage'])
        
        # Convert datetime strings back to datetime objects
        for dt_field in ['created_at', 'trained_at', 'validated_at', 'deployed_at', 'last_updated', 'expiry_date']:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        
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
            
                self.logger.info("✅ Enterprise model database initialized")
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
                monitoring_metrics=[],
                maintenance_schedule="As needed",
                expiry_date=None,
                security_scan_status="Not Scanned",
                compliance_checklist={},
                audit_trail=[{"timestamp": datetime.now().isoformat(), "event": "model_registered", "details": f"Model registered with name: {model_name}"}],
                access_permissions=["admin", "ml_engineer"]
            )
            
            # Save metadata and add to registry
            self._save_metadata(metadata)
            self._add_to_database(metadata)
            self.model_registry[model_id] = metadata
            
            self.logger.info(f"✅ New model registered successfully: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register new model: {e}")
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
                resource_requirements={},
                scaling_parameters={},
                health_check_url=None,
                monitoring_metrics=[],
                maintenance_schedule="As needed",
                expiry_date=None,
                security_scan_status="Not Scanned",
                compliance_checklist={"initial_registration": True},
                audit_trail=[{"timestamp": datetime.now().isoformat(), "event": "model_registered", "details": f"Model registered with name: {name}"}],
                access_permissions=["admin", "ml_engineer"]
            )
            
            # Save metadata and add to registry
            self._save_metadata(metadata)
            self._add_to_database(metadata)
            self.model_registry[model_id] = metadata
            
            self.logger.info(f"✅ New model registered successfully: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register new model: {e}")
            raise
    
    def log_training_session(self, 
                    model_id: str,
                            model_object: Any,
                            training_data: pd.DataFrame,
                            validation_data: pd.DataFrame,
                            training_history: Dict[str, Any],
                            duration_seconds: float) -> bool:
        """
        Log a training session for a registered model
        
        Args:
            model_id: Model identifier
            model_object: The trained model object
            training_data: Training dataset
            validation_data: Validation dataset
            training_history: Training history from Keras/PyTorch
            duration_seconds: Training duration
            
        Returns:
            Success status
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Save model to file
            model_path = self._save_model(model_id, model_object)
            
            # Update metadata
            metadata.status = ModelStatus.TRAINED
            metadata.trained_at = datetime.now()
            metadata.last_updated = datetime.now()
            metadata.training_duration_seconds = duration_seconds
            metadata.training_data_samples = len(training_data)
            metadata.validation_data_samples = len(validation_data)
            metadata.feature_count = training_data.shape[1] - 1
            metadata.model_file_path = str(model_path)
            metadata.file_size_bytes = model_path.stat().st_size
            metadata.file_hash_sha256 = self._calculate_file_hash(model_path)
            
            # Extract and update training metrics
            metadata.training_metrics = self._extract_metrics(training_history)
            
            # Log audit event
            self._log_audit_event(model_id, "model_trained", "Model training session completed")
            
            # Update registry and files
            self.model_registry[model_id] = metadata
            self._save_metadata(metadata)
            self._update_in_database(metadata)
            
            self.logger.info(f"✅ Training session logged for model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log training session: {e}")
            return False

    def validate_model_performance(self, 
                                  model_id: str, 
                                  validation_data: pd.DataFrame,
                                  validation_labels: pd.Series,
                                  predictions: np.ndarray) -> bool:
        """
        Validate model performance against enterprise standards
        
        Args:
            model_id: Model identifier
            validation_data: Validation dataset
            validation_labels: Validation labels
            predictions: Model predictions
            
        Returns:
            Validation status
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            # Calculate validation metrics
            validation_metrics = {
                'accuracy': accuracy_score(validation_labels, (predictions > 0.5).astype(int)),
                'roc_auc': roc_auc_score(validation_labels, predictions),
                'precision': precision_score(validation_labels, (predictions > 0.5).astype(int)),
                'recall': recall_score(validation_labels, (predictions > 0.5).astype(int)),
                'f1_score': f1_score(validation_labels, (predictions > 0.5).astype(int))
            }
            
            # Update metadata
            metadata.validation_metrics = validation_metrics
            metadata.actual_performance = validation_metrics['roc_auc']
            metadata.validated_at = datetime.now()
            metadata.last_updated = datetime.now()
            
            # Check against enterprise standards
            if metadata.actual_performance >= metadata.target_performance:
                metadata.status = ModelStatus.VALIDATED
                self.logger.info(f"✅ Model validation passed for: {model_id}")
                validation_passed = True
            else:
                metadata.status = ModelStatus.ERROR
                self.logger.warning(f"⚠️ Model validation failed for: {model_id}")
                validation_passed = False
            
            # Log audit event
            self._log_audit_event(model_id, "model_validated", f"Validation status: {'Passed' if validation_passed else 'Failed'}")
            
            # Update registry and files
            self.model_registry[model_id] = metadata
            self._save_metadata(metadata)
            self._update_in_database(metadata)
            
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate model performance: {e}")
            return False

    def promote_model_to_staging(self, model_id: str) -> bool:
        """Promote a validated model to staging"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            if metadata.status != ModelStatus.VALIDATED:
                raise Exception("Model must be validated before promotion")
            
            # Update deployment stage
            metadata.deployment_stage = DeploymentStage.STAGING
            metadata.last_updated = datetime.now()
            
            # Log audit event
            self._log_audit_event(model_id, "promoted_to_staging", "Model promoted to staging environment")
            
            # Update registry and files
            self.model_registry[model_id] = metadata
            self._save_metadata(metadata)
            self._update_in_database(metadata)
            
            self.logger.info(f"✅ Model promoted to staging: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to promote model to staging: {e}")
            return False
    
    def deploy_model_to_production(self, model_id: str) -> bool:
        """Deploy a model to production"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            if metadata.deployment_stage != DeploymentStage.STAGING:
                raise Exception("Model must be in staging before deployment")
            
            # Update status and deployment stage
            metadata.deployment_stage = DeploymentStage.PRODUCTION
            metadata.status = ModelStatus.ACTIVE
            metadata.deployed_at = datetime.now()
            metadata.last_updated = datetime.now()
            
            # Log audit event
            self._log_audit_event(model_id, "deployed_to_production", "Model deployed to production")
            
            # Update registry and files
            self.model_registry[model_id] = metadata
            self._save_metadata(metadata)
            self._update_in_database(metadata)
            
            self.logger.info(f"🚀 Model deployed to production: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy model to production: {e}")
            return False
    
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
    
    def get_model_metadata(self, model_id: str) -> Optional[EnterpriseModelMetadata]:
        """Get model metadata from registry"""
        return self.model_registry.get(model_id)

    def list_all_models(self) -> List[EnterpriseModelMetadata]:
        """List all registered models"""
        return list(self.model_registry.values())

    def get_best_model(self, model_type: ModelType, stage: DeploymentStage = DeploymentStage.PRODUCTION) -> Optional[EnterpriseModelMetadata]:
        """
        Get the best performing model of a specific type and deployment stage
        
        Args:
            model_type: Type of model
            stage: Deployment stage
            
        Returns:
            Best performing model metadata
        """
        try:
            models = [
                m for m in self.model_registry.values()
                if m.model_type == model_type and m.deployment_stage == stage
            ]
            
            if not models:
                return None
            
            return max(models, key=lambda m: m.actual_performance)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get best model: {e}")
            return None
    
    def _save_model(self, model_id: str, model_object: Any) -> Path:
        """Save model object to file"""
        try:
            model_path = self.models_dir / f"{model_id}.joblib"
            joblib.dump(model_object, model_path)
            self.logger.info(f"💾 Model saved to: {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def _save_metadata(self, metadata: EnterpriseModelMetadata):
        """Save metadata to JSON file"""
        try:
            with open(metadata.metadata_file_path, 'w') as f:
                # Custom JSON encoder for datetime and Enum objects
                def json_serializer(obj):
                    if isinstance(obj, (datetime, timedelta)):
                        return obj.isoformat()
                    if isinstance(obj, Enum):
                        return obj.value
                    raise TypeError(f"Type {type(obj)} not serializable")
                
                json.dump(metadata.to_dict(), f, indent=4, default=json_serializer)
            self.logger.info(f"📝 Metadata saved for model: {metadata.model_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_metrics(self, history: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from Keras training history"""
        metrics = {}
        for key, value in history.history.items():
            if isinstance(value, list):
                metrics[key] = value[-1]
        return metrics

    def _log_audit_event(self, model_id: str, event: str, details: str):
        """Log an audit event for a model"""
        try:
            metadata = self.model_registry[model_id]
            audit_event = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "details": details
            }
            metadata.audit_trail.append(audit_event)
            self.model_registry[model_id] = metadata
            self._update_in_database(metadata)
        except Exception as e:
            self.logger.error(f"❌ Failed to log audit event: {e}")
    
    def _add_to_database(self, metadata: EnterpriseModelMetadata):
        """Add new model metadata to SQLite database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
                # Prepare data for insertion
                data_dict = metadata.to_dict()
                
                # Convert enums and complex types to strings
                for key, value in data_dict.items():
                    if isinstance(value, Enum):
                        data_dict[key] = value.value
                    elif isinstance(value, (list, dict)):
                        data_dict[key] = json.dumps(value)
                    elif isinstance(value, datetime):
                        data_dict[key] = value.isoformat()
                
                columns = ', '.join(data_dict.keys())
                placeholders = ', '.join('?' * len(data_dict))
                sql = f"INSERT INTO enterprise_models ({columns}) VALUES ({placeholders})"
                
                cursor.execute(sql, list(data_dict.values()))
            conn.commit()
            
            self.logger.info(f"✅ Model added to database: {metadata.model_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to add model to database: {e}")
            raise
    
    def _update_in_database(self, metadata: EnterpriseModelMetadata):
        """Update existing model metadata in SQLite database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
                # Prepare data for update
                data_dict = metadata.to_dict()
                
                # Convert enums and complex types to strings
                for key, value in data_dict.items():
                    if isinstance(value, Enum):
                        data_dict[key] = value.value
                    elif isinstance(value, (list, dict)):
                        data_dict[key] = json.dumps(value)
                    elif isinstance(value, datetime):
                        data_dict[key] = value.isoformat()
                
                set_clause = ', '.join([f"{key} = ?" for key in data_dict.keys()])
                sql = f"UPDATE enterprise_models SET {set_clause} WHERE model_id = ?"
                
                values = list(data_dict.values()) + [metadata.model_id]
                cursor.execute(sql, values)
            conn.commit()
            
            self.logger.info(f"✅ Model updated in database: {metadata.model_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to update model in database: {e}")
            raise

    def _backup_model(self, model_id: str):
        """Create a backup of the model file"""
        if not self.enterprise_settings['auto_backup_enabled']:
                return
            
        try:
            metadata = self.model_registry[model_id]
            model_path = Path(metadata.model_file_path)
            
            backup_filename = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib.bak"
            backup_path = self.backups_dir / backup_filename
            
            shutil.copy2(model_path, backup_path)
            
            metadata.backup_paths.append(str(backup_path))
            self._update_in_database(metadata)
            
            self.logger.info(f"✅ Model backup created: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to backup model: {e}")

    def _restore_from_backup(self, model_id: str) -> Path:
        """Restore a model from the latest backup"""
        try:
            metadata = self.model_registry[model_id]
            if not metadata.backup_paths:
                raise FileNotFoundError("No backups available")
            
            latest_backup = sorted(metadata.backup_paths)[-1]
            model_path = Path(metadata.model_file_path)
            
            shutil.copy2(latest_backup, model_path)
            
            self.logger.info(f"✅ Model restored from backup: {latest_backup}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore model from backup: {e}")
            raise
    
# Factory function for creating enterprise model manager
def get_enterprise_model_manager(config: Dict[str, Any] = None, 
                                logger: logging.Logger = None) -> EnterpriseModelManager:
    """Create Enterprise Model Manager instance"""
    return EnterpriseModelManager(config=config, logger=logger)

# Export for use in other modules
__all__ = ['EnterpriseModelManager', 'EnterpriseModelMetadata', 'ModelType', 'ModelStatus', 
           'DeploymentStage', 'get_enterprise_model_manager']
