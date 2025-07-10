#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 ENHANCED CNN-LSTM ENGINE WITH ENTERPRISE MODEL MANAGEMENT
Enhanced CNN-LSTM Engine พร้อมระบบจัดการโมเดลระดับ Enterprise

🎯 New Enterprise Features:
✅ Automatic Model Registration & Versioning
✅ Model Performance Tracking & Validation
✅ Production-Ready Model Deployment Pipeline
✅ Model Backup & Recovery System
✅ Enterprise Security & Compliance
✅ Model Lifecycle Management

วันที่: 7 กรกฎาคม 2025
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
import json

# Environment variables for stable operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

# Import Enterprise Model Manager
try:
    from core.enterprise_model_manager_v2 import (
        EnterpriseModelManager, 
        ModelType, 
        ModelStatus,
        DeploymentStage
    )
    ENTERPRISE_MODEL_MANAGER_AVAILABLE = True
except ImportError:
    ENTERPRISE_MODEL_MANAGER_AVAILABLE = False

# TensorFlow imports with error handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Configure TensorFlow
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(42)
    
    # GPU Configuration
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Scikit-learn imports
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class EnterpriseCNNLSTMEngine:
    """
    🏢 Enterprise CNN-LSTM Engine with Model Management
    Production-Grade CNN-LSTM with Enterprise Features
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 logger: logging.Logger = None,
                 model_manager: EnterpriseModelManager = None):
        """Initialize Enterprise CNN-LSTM Engine"""
        self.config = config or {}
        self.logger = logger or get_unified_logger()
        
        # Initialize Model Manager
        if ENTERPRISE_MODEL_MANAGER_AVAILABLE:
            self.model_manager = model_manager or EnterpriseModelManager(
                config=self.config,
                logger=self.logger
            )
        else:
            self.model_manager = None
            self.logger.warning("⚠️ Enterprise Model Manager not available")
        
        # Model configuration
        self.model_config = {
            'sequence_length': self.config.get('sequence_length', 60),
            'cnn_filters': self.config.get('cnn_filters', 64),
            'cnn_kernel_size': self.config.get('cnn_kernel_size', 3),
            'lstm_units': self.config.get('lstm_units', 100),
            'dropout_rate': self.config.get('dropout_rate', 0.2),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'batch_size': self.config.get('batch_size', 32),
            'epochs': self.config.get('epochs', 100),
            'early_stopping_patience': self.config.get('early_stopping_patience', 10),
            'validation_split': self.config.get('validation_split', 0.2)
        }
        
        # Model state
        self.model = None
        self.scaler = None
        self.training_history = None
        self.current_model_id = None
        self.is_trained = False
        
        # Enterprise settings
        self.enterprise_settings = {
            'target_auc': 0.70,
            'min_accuracy': 0.65,
            'enable_model_versioning': True,
            'auto_backup': True,
            'performance_monitoring': True,
            'compliance_validation': True
        }
        
        self.logger.info("🏢 Enterprise CNN-LSTM Engine initialized")
    
    def create_model_architecture(self, input_shape: Tuple[int, int]) -> 'tf.keras.Model':
        """
        Create CNN-LSTM model architecture for Elliott Wave pattern recognition
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled TensorFlow model
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                raise Exception("TensorFlow not available")
            
            self.logger.info("🏗️ Creating CNN-LSTM model architecture...")
            
            model = Sequential([
                # CNN Layer for Pattern Recognition
                Conv1D(filters=self.model_config['cnn_filters'],
                      kernel_size=self.model_config['cnn_kernel_size'],
                      activation='relu',
                      input_shape=input_shape),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rate']),
                
                # Additional CNN Layer
                Conv1D(filters=self.model_config['cnn_filters'] // 2,
                      kernel_size=self.model_config['cnn_kernel_size'],
                      activation='relu'),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rate']),
                
                # LSTM Layer for Sequence Learning
                LSTM(units=self.model_config['lstm_units'],
                     return_sequences=True,
                     dropout=self.model_config['dropout_rate'],
                     recurrent_dropout=self.model_config['dropout_rate']),
                BatchNormalization(),
                
                # Second LSTM Layer
                LSTM(units=self.model_config['lstm_units'] // 2,
                     dropout=self.model_config['dropout_rate'],
                     recurrent_dropout=self.model_config['dropout_rate']),
                BatchNormalization(),
                
                # Dense Layers
                Dense(64, activation='relu'),
                Dropout(self.model_config['dropout_rate']),
                Dense(32, activation='relu'),
                Dropout(self.model_config['dropout_rate']),
                
                # Output Layer (Binary Classification)
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.model_config['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info(f"✅ CNN-LSTM model created with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create model architecture: {e}")
            raise
    
    def prepare_data_sequences(self, 
                             data: pd.DataFrame, 
                             target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for CNN-LSTM training
        
        Args:
            data: Input dataframe with features
            target_column: Name of target column
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        try:
            self.logger.info("📊 Preparing sequential data for CNN-LSTM...")
            
            # Separate features and target
            feature_columns = [col for col in data.columns if col != target_column]
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Scale features
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Create sequences
            sequence_length = self.model_config['sequence_length']
            X_sequences, y_sequences = [], []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
                y_sequences.append(y[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            self.logger.info(f"✅ Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data sequences: {e}")
            raise
    
    def train_model(self, 
                   training_data: pd.DataFrame,
                   target_column: str,
                   validation_data: pd.DataFrame = None,
                   model_name: str = "CNN_LSTM_Elliott_Wave",
                   business_purpose: str = "Elliott Wave Pattern Recognition") -> Dict[str, Any]:
        """
        Train CNN-LSTM model with enterprise management
        
        Args:
            training_data: Training dataset
            target_column: Target column name
            validation_data: Optional validation dataset
            model_name: Name for the model
            business_purpose: Business purpose description
            
        Returns:
            Training results and model metadata
        """
        try:
            self.logger.info("🚀 Starting Enterprise CNN-LSTM Training...")
            
            # Register new model in enterprise system
            if self.model_manager:
                self.current_model_id = self.model_manager.register_new_model(
                    model_name=model_name,
                    model_type=ModelType.CNN_LSTM,
                    business_purpose=business_purpose,
                    use_case_description="Deep learning model for Elliott Wave pattern recognition in financial time series",
                    training_config=self.model_config
                )
                self.logger.info(f"📝 Model registered: {self.current_model_id}")
            
            # Update model status to training
            if self.model_manager:
                self.model_manager.update_model_status(
                    self.current_model_id, 
                    ModelStatus.TRAINING
                )
            
            training_start_time = datetime.now()
            
            # Prepare training data
            X_train, y_train = self.prepare_data_sequences(training_data, target_column)
            
            # Prepare validation data if provided
            if validation_data is not None:
                X_val, y_val = self.prepare_data_sequences(validation_data, target_column)
            else:
                # Split training data for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, 
                    test_size=self.model_config['validation_split'],
                    random_state=42,
                    stratify=y_train
                )
            
            # Create model architecture
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.create_model_architecture(input_shape)
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_config['early_stopping_patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            self.logger.info("🎯 Training CNN-LSTM model...")
            self.training_history = self.model.fit(
                X_train, y_train,
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_predictions_binary = (train_predictions > 0.5).astype(int)
            
            training_metrics = {
                'accuracy': accuracy_score(y_train, train_predictions_binary),
                'precision': precision_score(y_train, train_predictions_binary, average='weighted', zero_division=0),
                'recall': recall_score(y_train, train_predictions_binary, average='weighted', zero_division=0),
                'f1_score': f1_score(y_train, train_predictions_binary, average='weighted', zero_division=0),
                'auc': roc_auc_score(y_train, train_predictions) if len(np.unique(y_train)) > 1 else 0.0,
                'loss': float(self.training_history.history['loss'][-1]),
                'val_loss': float(self.training_history.history['val_loss'][-1])
            }
            
            # Calculate validation metrics
            val_predictions = self.model.predict(X_val)
            val_predictions_binary = (val_predictions > 0.5).astype(int)
            
            validation_metrics = {
                'accuracy': accuracy_score(y_val, val_predictions_binary),
                'precision': precision_score(y_val, val_predictions_binary, average='weighted', zero_division=0),
                'recall': recall_score(y_val, val_predictions_binary, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, val_predictions_binary, average='weighted', zero_division=0),
                'auc': roc_auc_score(y_val, val_predictions) if len(np.unique(y_val)) > 1 else 0.0
            }
            
            # Update model with training results
            if self.model_manager:
                # Save model to enterprise system
                model_file_path = self.model_manager.save_trained_model(
                    model_id=self.current_model_id,
                    model_object=self.model,
                    scaler=self.scaler,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    training_duration=training_duration,
                    training_samples=len(X_train),
                    validation_samples=len(X_val),
                    feature_count=X_train.shape[2]
                )
                
                self.logger.info(f"💾 Model saved: {model_file_path}")
            
            # Mark as trained
            self.is_trained = True
            
            # Prepare results
            results = {
                'model_id': self.current_model_id,
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics,
                'training_duration_seconds': training_duration,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': X_train.shape[2],
                'model_parameters': self.model.count_params() if self.model else 0,
                'enterprise_compliant': validation_metrics['auc'] >= self.enterprise_settings['target_auc']
            }
            
            # Log training completion
            auc_score = validation_metrics['auc']
            if auc_score >= self.enterprise_settings['target_auc']:
                self.logger.info(f"🎉 Training completed successfully! AUC: {auc_score:.4f} (Target: {self.enterprise_settings['target_auc']})")
            else:
                self.logger.warning(f"⚠️ Training completed but AUC below target: {auc_score:.4f} (Target: {self.enterprise_settings['target_auc']})")
            
            return results
            
        except Exception as e:
            # Update model status to failed
            if self.model_manager and self.current_model_id:
                self.model_manager.update_model_status(
                    self.current_model_id, 
                    ModelStatus.FAILED
                )
            
            self.logger.error(f"❌ CNN-LSTM training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            data: Input data for prediction
            target_column: Target column to exclude from features
            
        Returns:
            Prediction probabilities
        """
        try:
            if not self.is_trained or self.model is None:
                raise Exception("Model is not trained")
            
            self.logger.info("🔮 Making predictions with CNN-LSTM model...")
            
            # Prepare data for prediction
            if target_column and target_column in data.columns:
                feature_columns = [col for col in data.columns if col != target_column]
                X = data[feature_columns].values
            else:
                X = data.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create sequences
            sequence_length = self.model_config['sequence_length']
            X_sequences = []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
            
            if not X_sequences:
                raise Exception(f"Not enough data for sequences. Need at least {sequence_length} samples.")
            
            X_sequences = np.array(X_sequences)
            
            # Make predictions
            predictions = self.model.predict(X_sequences)
            
            self.logger.info(f"✅ Generated {len(predictions)} predictions")
            
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def load_model_from_enterprise(self, model_id: str) -> bool:
        """
        Load model from enterprise system
        
        Args:
            model_id: Enterprise model identifier
            
        Returns:
            Success status
        """
        try:
            if not self.model_manager:
                raise Exception("Enterprise Model Manager not available")
            
            self.logger.info(f"📥 Loading model from enterprise system: {model_id}")
            
            # Load model from enterprise system
            model_data = self.model_manager.load_model(model_id)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.current_model_id = model_id
            self.is_trained = True
            
            self.logger.info(f"✅ Model loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        try:
            if not self.model_manager or not self.current_model_id:
                return {"error": "No model loaded or model manager not available"}
            
            return self.model_manager.get_model_performance_summary(self.current_model_id)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def validate_enterprise_compliance(self) -> Dict[str, Any]:
        """Validate model compliance with enterprise standards"""
        try:
            if not self.is_trained:
                return {"compliant": False, "reason": "Model not trained"}
            
            if not self.model_manager or not self.current_model_id:
                return {"compliant": False, "reason": "Enterprise management not available"}
            
            # Get model metadata
            metadata = self.model_manager.model_registry.get(self.current_model_id)
            if not metadata:
                return {"compliant": False, "reason": "Model metadata not found"}
            
            # Check compliance criteria
            validation_auc = metadata.validation_metrics.get('auc', 0.0)
            target_auc = self.enterprise_settings['target_auc']
            
            compliance_checks = {
                'auc_requirement': validation_auc >= target_auc,
                'model_registered': True,
                'validation_completed': len(metadata.validation_metrics) > 0,
                'backup_created': len(metadata.backup_paths) > 0,
                'security_scan': metadata.security_scan_status == 'passed'
            }
            
            overall_compliant = all(compliance_checks.values())
            
            return {
                'compliant': overall_compliant,
                'checks': compliance_checks,
                'auc_score': validation_auc,
                'target_auc': target_auc,
                'model_id': self.current_model_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Compliance validation failed: {e}")
            return {"compliant": False, "error": str(e)}

# Factory function for creating enterprise CNN-LSTM engine
def create_enterprise_cnn_lstm_engine(config: Dict[str, Any] = None, 
                                     logger: logging.Logger = None) -> EnterpriseCNNLSTMEngine:
    """Create Enterprise CNN-LSTM Engine instance"""
    return EnterpriseCNNLSTMEngine(config=config, logger=logger)

# Export for use in other modules
__all__ = ['EnterpriseCNNLSTMEngine', 'create_enterprise_cnn_lstm_engine']
