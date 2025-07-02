#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 OPTIMIZED MENU 1: ELLIOTT WAVE SYSTEM
Resource-Optimized Elliott Wave Pipeline with Error Handling

Key Improvements:
- Reduced resource usage
- Better error handling
- Simplified progress tracking
- Conservative ML settings
- Automatic fallback mechanisms
"""

import sys
import os
import time
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Suppress warnings early
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Essential imports
import pandas as pd
import numpy as np

# Project imports
sys.path.append(str(Path(__file__).parent.parent))

class OptimizedMenu1ElliottWave:
    """
    🌊 Optimized Elliott Wave Menu - Resource Efficient
    ระบบ Elliott Wave ที่ปรับปรุงให้ใช้ทรัพยากรน้อย
    """
    
    def __init__(self, config: Dict = None, logger=None, resource_manager=None):
        """Initialize with conservative settings"""
        self.config = config or {}
        self.logger = logger or logging.getLogger("OptimizedMenu1")
        self.resource_manager = resource_manager
        
        # Conservative settings
        self.max_data_rows = 10000  # Limit data size
        self.max_features = 15      # Limit feature count
        self.max_iterations = 50    # Limit ML iterations
        
        # Status tracking
        self.status = {
            'initialized': False,
            'data_loaded': False,
            'features_created': False,
            'model_trained': False,
            'completed': False,
            'errors': [],
            'warnings': []
        }
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🌊 Optimized Elliott Wave Menu initialized")
    
    def _initialize_components(self):
        """Initialize required components with error handling"""
        try:
            # Try to import and initialize data processor
            from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
            self.data_processor = ElliottWaveDataProcessor(logger=self.logger)
            
            # Initialize other components as needed
            self.ml_components = {}
            self.status['initialized'] = True
            
        except ImportError as e:
            error_msg = f"Failed to initialize components: {e}"
            self.status['errors'].append(error_msg)
            self.logger.error(error_msg)
            
            # Create fallback data processor
            self.data_processor = None
        except Exception as e:
            error_msg = f"Unexpected error during initialization: {e}"
            self.status['errors'].append(error_msg)
            self.logger.error(error_msg)
    
    def run_optimized_pipeline(self) -> Dict[str, Any]:
        """
        Run the optimized Elliott Wave pipeline
        Returns results with minimal resource usage
        """
        results = {
            'success': False,
            'message': '',
            'data': {},
            'performance': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            self.logger.info("🚀 Starting Optimized Elliott Wave Pipeline")
            
            # Step 1: Load and validate data (with limits)
            self.logger.info("📊 Step 1: Loading data with conservative limits...")
            data_result = self._load_data_conservatively()
            if not data_result['success']:
                results['errors'].extend(data_result['errors'])
                return results
            
            # Step 2: Create basic features (limited set)
            self.logger.info("⚙️ Step 2: Creating essential features...")
            features_result = self._create_essential_features(data_result['data'])
            if not features_result['success']:
                results['errors'].extend(features_result['errors'])
                return results
            
            # Step 3: Basic feature selection (simplified)
            self.logger.info("🎯 Step 3: Selecting top features...")
            selection_result = self._select_top_features(features_result['data'])
            if not selection_result['success']:
                results['warnings'].extend(selection_result['warnings'])
                # Continue with all features if selection fails
                final_features = features_result['data']
            else:
                final_features = selection_result['data']
            
            # Step 4: Simple model training (conservative)
            self.logger.info("🧠 Step 4: Training simplified model...")
            model_result = self._train_simple_model(final_features)
            if not model_result['success']:
                results['errors'].extend(model_result['errors'])
                return results
            
            # Step 5: Performance evaluation
            self.logger.info("📈 Step 5: Evaluating performance...")
            eval_result = self._evaluate_performance(model_result['data'])
            
            # Compile final results
            results.update({
                'success': True,
                'message': 'Optimized pipeline completed successfully',
                'data': {
                    'data_rows': len(data_result['data']),
                    'feature_count': len(final_features.columns) - 1,  # Exclude target
                    'model_type': model_result['data'].get('model_type', 'SimpleModel'),
                    'performance': eval_result.get('metrics', {}),
                    'timestamp': datetime.now().isoformat()
                },
                'performance': self._get_resource_usage(),
                'errors': results['errors'],
                'warnings': results['warnings']
            })
            
            self.logger.info("✅ Optimized Elliott Wave Pipeline completed successfully!")
            
        except Exception as e:
            error_msg = f"Pipeline execution error: {str(e)}"
            results['errors'].append(error_msg)
            results['message'] = error_msg
            self.logger.error(error_msg)
        
        return results
    
    def _load_data_conservatively(self) -> Dict[str, Any]:
        """Load data with conservative memory usage"""
        try:
            if not self.data_processor:
                return {
                    'success': False,
                    'errors': ['Data processor not available'],
                    'data': None
                }
            
            # Load data with row limit
            data = self.data_processor.load_real_data()
            if data is None or len(data) == 0:
                return {
                    'success': False,
                    'errors': ['No data loaded'],
                    'data': None
                }
            
            # Limit data size for memory efficiency
            if len(data) > self.max_data_rows:
                data = data.tail(self.max_data_rows)
                self.logger.info(f"📊 Limited data to {self.max_data_rows} rows for efficiency")
            
            self.status['data_loaded'] = True
            return {
                'success': True,
                'data': data,
                'message': f'Loaded {len(data)} rows'
            }
            
        except Exception as e:
            error_msg = f"Data loading error: {str(e)}"
            return {
                'success': False,
                'errors': [error_msg],
                'data': None
            }
    
    def _create_essential_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create essential features with memory efficiency"""
        try:
            if self.data_processor:
                # Use data processor to create features
                features = self.data_processor.create_elliott_wave_features(data)
            else:
                # Fallback: create basic features manually
                features = self._create_basic_features_fallback(data)
            
            if features is None or len(features) == 0:
                return {
                    'success': False,
                    'errors': ['No features created'],
                    'data': None
                }
            
            # Limit feature count
            if len(features.columns) > self.max_features + 1:  # +1 for target
                # Keep only numeric columns and limit count
                numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
                if 'target' in features.columns:
                    keep_cols = ['target'] + numeric_cols[:self.max_features]
                else:
                    keep_cols = numeric_cols[:self.max_features]
                
                features = features[keep_cols]
                self.logger.info(f"⚙️ Limited features to {len(features.columns)} for efficiency")
            
            self.status['features_created'] = True
            return {
                'success': True,
                'data': features,
                'message': f'Created {len(features.columns)} features'
            }
            
        except Exception as e:
            error_msg = f"Feature creation error: {str(e)}"
            return {
                'success': False,
                'errors': [error_msg],
                'data': None
            }
    
    def _create_basic_features_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features as fallback"""
        try:
            features = data.copy()
            
            # Basic price features
            if 'close' in features.columns:
                features['price_change'] = features['close'].pct_change()
                features['price_sma_5'] = features['close'].rolling(5).mean()
                features['price_sma_10'] = features['close'].rolling(10).mean()
                features['price_std_5'] = features['close'].rolling(5).std()
            
            # Basic volume features (if available)
            if 'volume' in features.columns:
                features['volume_change'] = features['volume'].pct_change()
                features['volume_sma_5'] = features['volume'].rolling(5).mean()
            
            # Create simple target (price direction)
            if 'close' in features.columns:
                features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
            
            # Remove rows with NaN
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Basic feature creation error: {e}")
            return None
    
    def _select_top_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Select top features using simple correlation"""
        try:
            if 'target' not in data.columns:
                return {
                    'success': False,
                    'warnings': ['No target column for feature selection'],
                    'data': data
                }
            
            # Simple correlation-based feature selection
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_features:
                numeric_features.remove('target')
            
            if len(numeric_features) <= self.max_features:
                return {
                    'success': True,
                    'data': data,
                    'message': f'All {len(numeric_features)} features selected'
                }
            
            # Calculate correlation with target
            correlations = data[numeric_features].corrwith(data['target']).abs()
            top_features = correlations.nlargest(self.max_features).index.tolist()
            
            # Create final dataset
            final_features = data[top_features + ['target']]
            
            return {
                'success': True,
                'data': final_features,
                'message': f'Selected top {len(top_features)} features'
            }
            
        except Exception as e:
            error_msg = f"Feature selection error: {str(e)}"
            return {
                'success': False,
                'warnings': [error_msg],
                'data': data
            }
    
    def _train_simple_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train a simple model with conservative settings"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # Prepare data
            if 'target' not in data.columns:
                return {
                    'success': False,
                    'errors': ['No target column for model training'],
                    'data': None
                }
            
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train simple model
            model = RandomForestClassifier(
                n_estimators=10,  # Very few trees
                max_depth=5,      # Shallow trees
                random_state=42,
                n_jobs=1          # Single thread
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.status['model_trained'] = True
            
            return {
                'success': True,
                'data': {
                    'model': model,
                    'model_type': 'RandomForest',
                    'accuracy': accuracy,
                    'auc': auc,
                    'feature_names': X.columns.tolist(),
                    'n_samples_train': len(X_train),
                    'n_samples_test': len(X_test)
                },
                'message': f'Model trained: Accuracy={accuracy:.3f}, AUC={auc:.3f}'
            }
            
        except Exception as e:
            error_msg = f"Model training error: {str(e)}"
            return {
                'success': False,
                'errors': [error_msg],
                'data': None
            }
    
    def _evaluate_performance(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            if not model_data:
                return {'metrics': {}}
            
            metrics = {
                'accuracy': model_data.get('accuracy', 0.0),
                'auc': model_data.get('auc', 0.0),
                'model_type': model_data.get('model_type', 'Unknown'),
                'feature_count': len(model_data.get('feature_names', [])),
                'train_samples': model_data.get('n_samples_train', 0),
                'test_samples': model_data.get('n_samples_test', 0)
            }
            
            # Determine performance grade
            auc = metrics['auc']
            if auc >= 0.7:
                grade = 'A'
            elif auc >= 0.6:
                grade = 'B'
            elif auc >= 0.5:
                grade = 'C'
            else:
                grade = 'D'
            
            metrics['performance_grade'] = grade
            
            self.status['completed'] = True
            
            return {'metrics': metrics}
            
        except Exception as e:
            self.logger.error(f"Performance evaluation error: {e}")
            return {'metrics': {}}
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            if self.resource_manager:
                return self.resource_manager.get_current_performance()
            else:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'memory_mb': round(memory.used / (1024**2), 1),
                    'memory_percent': round(memory.percent, 1),
                    'cpu_percent': psutil.cpu_percent()
                }
        except Exception:
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current menu status"""
        return {
            'status': self.status.copy(),
            'resource_usage': self._get_resource_usage(),
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self) -> Dict[str, Any]:
        """
        🚀 Main run method for menu compatibility
        Calls the optimized pipeline
        """
        try:
            self.logger.info("🚀 Starting Menu 1: Optimized Elliott Wave Full Pipeline")
            
            # Run the optimized pipeline
            result = self.run_optimized_pipeline()
            
            if result['success']:
                self.logger.info("✅ Optimized Elliott Wave Pipeline completed successfully")
            else:
                self.logger.error("❌ Optimized Elliott Wave Pipeline failed")
                if result['errors']:
                    for error in result['errors']:
                        self.logger.error(f"   Error: {error}")
            
            return result
            
        except Exception as e:
            error_msg = f"Menu 1 run failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'errors': [error_msg]
            }

# Menu interface function
def menu_1_elliott_wave_optimized(config: Dict = None, logger=None, resource_manager=None) -> Dict[str, Any]:
    """
    Optimized Elliott Wave Menu 1 entry point
    """
    try:
        menu = OptimizedMenu1ElliottWave(config, logger, resource_manager)
        return menu.run_optimized_pipeline()
    except Exception as e:
        logger = logger or logging.getLogger("Menu1")
        logger.error(f"Menu 1 execution error: {e}")
        return {
            'success': False,
            'message': f'Menu 1 failed: {str(e)}',
            'errors': [str(e)]
        }

# Compatibility class for existing integration
class Menu1ElliottWave:
    """Compatibility wrapper for existing code"""
    
    def __init__(self, config: Dict = None, logger=None, resource_manager=None):
        self.optimized_menu = OptimizedMenu1ElliottWave(config, logger, resource_manager)
        self.logger = logger or logging.getLogger("Menu1")
    
    def run_elliott_wave_pipeline(self) -> Dict[str, Any]:
        """Run the optimized pipeline"""
        return self.optimized_menu.run_optimized_pipeline()
    
    def get_status(self) -> Dict[str, Any]:
        """Get menu status"""
        return self.optimized_menu.get_status()
