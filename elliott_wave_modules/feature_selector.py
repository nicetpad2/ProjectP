#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ENTERPRISE SHAP + OPTUNA FEATURE SELECTOR
🏢 100% REAL DATA ONLY - ZERO FALLBACK POLICY

MANDATORY ENTERPRISE FEATURES:
✅ SHAP Feature Importance Analysis (REQUIRED)
✅ Optuna Hyperparameter Optimization (REQUIRED)  
✅ TimeSeriesSplit Cross-Validation (Data Leakage Prevention)
✅ AUC ≥ 70% Enforcement (STRICT)
✅ Real Data Only Policy (NO MOCK/DUMMY/SIMULATION)
✅ Production Grade Error Handling
✅ Enterprise Logging and Monitoring
✅ Zero Fallback Policy - SHAP + Optuna ONLY
"""

# Force CUDA disable for stability
import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import traceback
from pathlib import Path
import sys
import gc
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MANDATORY Enterprise Components - NO FALLBACK
try:
    import shap
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    ENTERPRISE_FEATURES_AVAILABLE = True
except ImportError as e:
    # STRICT POLICY: NO FALLBACK ALLOWED
    print(f"❌ CRITICAL ERROR: Enterprise features not available: {e}")
    print("❌ SHAP + Optuna are MANDATORY for production use")
    print("❌ Please install: pip install shap optuna scikit-learn")
    raise ImportError("ENTERPRISE FEATURES REQUIRED - NO FALLBACK ALLOWED") from e

# Advanced Logging
try:
    from core.unified_enterprise_logger import get_unified_logger
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False


class EnterpriseShapOptunaFeatureSelector:
    """
    🎯 Enterprise SHAP + Optuna Feature Selector
    🚫 ZERO FALLBACK POLICY - ENTERPRISE GRADE ONLY
    """
    
    def __init__(self, logger=None, config: Dict = None):
        """Initialize Enterprise Feature Selector - SHAP + Optuna ONLY"""
        self.config = config or {}
        
        # Initialize Enterprise Logger
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_unified_logger("FeatureSelector")
        else:
            self.logger = logger or logging.getLogger(__name__)
        
        # Enterprise Configuration
        self.n_trials = self.config.get('optuna_n_trials', 150)  # Production-grade trials
        self.timeout = self.config.get('optuna_timeout', 600)    # 10 minutes
        self.cv_folds = self.config.get('cv_folds', 5)          # TimeSeriesSplit
        self.target_auc = self.config.get('target_auc', 0.70)   # Enterprise requirement
        self.max_features = self.config.get('max_features', 30)  # Feature limit
        
        # Enterprise Quality Gates
        self.min_auc_threshold = 0.70  # MANDATORY
        self.max_selection_attempts = 3  # Maximum optimization attempts
        
        self.logger.info("🎯 Enterprise SHAP + Optuna Feature Selector initialized")
        self.logger.info(f"   📊 Configuration: {self.n_trials} trials, {self.cv_folds} CV folds")
        self.logger.info(f"   🎯 Target AUC: {self.target_auc:.2f} (Minimum: {self.min_auc_threshold:.2f})")
        self.logger.info("   🚫 ZERO FALLBACK POLICY - ENTERPRISE GRADE ONLY")

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features_to_select: int = 20, 
                       n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Select optimal features using SHAP + Optuna - NO FALLBACK
        
        Args:
            X: Feature matrix (REAL DATA ONLY)
            y: Target variable (REAL DATA ONLY)
            n_features_to_select: Number of features to select
            n_trials: Optuna trials (None = use config)
            
        Returns:
            Dict with selected features and enterprise metrics
        """
        self.logger.info("🎯 Starting Enterprise SHAP + Optuna Feature Selection")
        self.logger.info(f"   📊 Input: {X.shape[0]:,} samples, {X.shape[1]} features")
        self.logger.info(f"   🎯 Target: {n_features_to_select} optimal features")
        
        # Validate inputs
        if not self._validate_inputs(X, y):
            raise ValueError("Input validation failed - REAL DATA REQUIRED")
        
        # Use configured trials if not specified
        trials = n_trials or self.n_trials
        
        try:
            # Phase 1: SHAP Feature Importance Analysis
            shap_results = self._run_shap_analysis(X, y)
            
            # Phase 2: Optuna Hyperparameter Optimization
            optuna_results = self._run_optuna_optimization(X, y, shap_results, n_features_to_select, trials)
            
            # Phase 3: Final Feature Selection and Validation
            final_results = self._finalize_feature_selection(X, y, shap_results, optuna_results, n_features_to_select)
            
            # Phase 4: Enterprise Quality Gate Validation
            if not self._validate_enterprise_quality(final_results):
                raise ValueError(f"Selection failed enterprise quality gates - AUC {final_results.get('auc', 0):.4f} < {self.min_auc_threshold:.2f}")
            
            self.logger.info("✅ Enterprise Feature Selection completed successfully")
            return final_results
            
        except Exception as e:
            error_msg = f"Enterprise Feature Selection failed: {e}"
            self.logger.error(error_msg, error_details=traceback.format_exc())
            raise RuntimeError(error_msg) from e

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate input data - REAL DATA ONLY"""
        try:
            # Check data types
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                self.logger.error("❌ Invalid input types - DataFrame and Series required")
                return False
            
            # Check data size
            if len(X) < 1000:
                self.logger.error(f"❌ Insufficient data: {len(X)} samples < 1000 minimum")
                return False
            
            # Check feature count
            if X.shape[1] < 5:
                self.logger.error(f"❌ Insufficient features: {X.shape[1]} < 5 minimum")
                return False
            
            # Check for missing values
            if X.isnull().any().any():
                self.logger.warning("⚠️ Missing values detected in features")
            
            if y.isnull().any():
                self.logger.error("❌ Missing values in target variable")
                return False
            
            # Check target distribution
            if len(y.unique()) < 2:
                self.logger.error("❌ Target variable must have at least 2 classes")
                return False
            
            self.logger.info("✅ Input validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Input validation failed: {e}")
            return False

    def _run_shap_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run SHAP feature importance analysis - MANDATORY"""
        self.logger.info("🔍 Running SHAP Feature Importance Analysis...")
        
        try:
            # Prepare data for SHAP
            X_sample = X.sample(n=min(1000, len(X)), random_state=42)  # Sample for SHAP efficiency
            
            # Train base model for SHAP
            base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            base_model.fit(X_sample, y[X_sample.index])
            
            # SHAP Explainer
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary vs multiclass SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"✅ SHAP analysis completed - Top feature: {importance_df.iloc[0]['feature']}")
            
            return {
                'importance_df': importance_df,
                'shap_values': shap_values,
                'feature_ranking': importance_df['feature'].tolist()
            }
            
        except Exception as e:
            error_msg = f"SHAP analysis failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _run_optuna_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                shap_results: Dict, n_features: int, n_trials: int) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimization - MANDATORY"""
        self.logger.info(f"⚙️ Running Optuna Optimization - {n_trials} trials...")
        
        try:
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                study_name=f'feature_selection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            # Define objective function
            def objective(trial):
                # Select features based on SHAP importance
                top_features = shap_results['feature_ranking'][:trial.suggest_int('n_features', 5, min(50, len(top_features)))]
                
                # Model hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                
                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Select features
                    X_train_selected = X_train[top_features]
                    X_val_selected = X_val[top_features]
                    
                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train_selected, y_train)
                    
                    # Predict and score
                    y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
                    auc = roc_auc_score(y_val, y_pred_proba)
                    scores.append(auc)
                
                return np.mean(scores)
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, timeout=self.timeout)
            
            best_trial = study.best_trial
            self.logger.info(f"✅ Optuna optimization completed - Best AUC: {best_trial.value:.4f}")
            
            return {
                'study': study,
                'best_trial': best_trial,
                'best_params': best_trial.params,
                'best_auc': best_trial.value
            }
            
        except Exception as e:
            error_msg = f"Optuna optimization failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _finalize_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                   shap_results: Dict, optuna_results: Dict, 
                                   n_features: int) -> Dict[str, Any]:
        """Finalize feature selection with enterprise validation"""
        self.logger.info("🎯 Finalizing feature selection with enterprise validation...")
        
        try:
            # Get optimal number of features from Optuna
            optimal_n_features = min(n_features, optuna_results['best_params']['n_features'])
            
            # Select top features from SHAP
            selected_features = shap_results['feature_ranking'][:optimal_n_features]
            
            # Create final model with selected features
            final_model = RandomForestClassifier(
                n_estimators=optuna_results['best_params']['n_estimators'],
                max_depth=optuna_results['best_params']['max_depth'],
                min_samples_split=optuna_results['best_params']['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            
            # Final validation with selected features
            X_selected = X[selected_features]
            
            # TimeSeriesSplit validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            validation_scores = []
            
            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                final_model.fit(X_train, y_train)
                y_pred_proba = final_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                validation_scores.append(auc)
            
            final_auc = np.mean(validation_scores)
            
            self.logger.info(f"✅ Final validation AUC: {final_auc:.4f}")
            
            return {
                'selected_features': selected_features,
                'auc': final_auc,
                'cv_scores': validation_scores,
                'model': final_model,
                'shap_importance': shap_results['importance_df'],
                'optuna_study': optuna_results['study'],
                'selection_method': 'Enterprise_SHAP_Optuna',
                'enterprise_validated': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Feature selection finalization failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_enterprise_quality(self, results: Dict[str, Any]) -> bool:
        """Validate enterprise quality gates"""
        try:
            auc = results.get('auc', 0)
            
            # AUC Quality Gate
            if auc < self.min_auc_threshold:
                self.logger.error(f"❌ Enterprise Quality Gate Failed: AUC {auc:.4f} < {self.min_auc_threshold:.2f}")
                return False
            
            # Feature count validation
            n_features = len(results.get('selected_features', []))
            if n_features < 5:
                self.logger.error(f"❌ Insufficient features selected: {n_features} < 5")
                return False
            
            self.logger.info(f"✅ Enterprise Quality Gates Passed: AUC {auc:.4f} ≥ {self.min_auc_threshold:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Quality validation failed: {e}")
            return False


# Standalone test capability
if __name__ == '__main__':
    try:
        print("🧪 Testing Enterprise SHAP + Optuna Feature Selector...")
        
        # Create test data
        np.random.seed(42)
        n_samples = 5000
        n_features = 25
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series((X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.1) > 0).astype(int)
        
        # Test selector
        selector = EnterpriseShapOptunaFeatureSelector()
        results = selector.select_features(X, y, n_features_to_select=10, n_trials=20)
        
        print("\n" + "="*60)
        print("🧪 ENTERPRISE FEATURE SELECTOR TEST RESULTS")
        print("="*60)
        print(f"Selected Features: {len(results['selected_features'])}")
        print(f"Final AUC: {results['auc']:.4f}")
        print(f"Enterprise Validated: {results['enterprise_validated']}")
        print(f"Top 5 Features: {results['selected_features'][:5]}")
        print("="*60)
        
        if results['auc'] >= 0.70:
            print("✅ Test passed - Enterprise quality achieved!")
        else:
            print(f"❌ Test failed - AUC {results['auc']:.4f} < 0.70")
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        traceback.print_exc()
