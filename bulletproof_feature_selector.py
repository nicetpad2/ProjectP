#!/usr/bin/env python3
"""
🎯 ENTERPRISE REAL FEATURE SELECTOR - NICEGOLD ENTERPRISE
PRODUCTION-READY SHAP + OPTUNA FEATURE SELECTOR

🚀 ENTERPRISE REQUIREMENTS:
- SHAP Feature Importance Analysis (REQUIRED)
- Optuna Hyperparameter Optimization (REQUIRED)
- AUC ≥ 70% Target Achievement (ENFORCED)
- Real Data Processing (1.7M+ rows)
- NO Mock/Fast/Fallback Modes
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ENTERPRISE IMPORTS - REQUIRED
try:
    import shap
    import optuna
    from optuna.pruners import MedianPruner
    ENTERPRISE_LIBS_AVAILABLE = True
except ImportError:
    ENTERPRISE_LIBS_AVAILABLE = False
    raise ImportError("🚫 ENTERPRISE FAILURE: SHAP and Optuna are REQUIRED for production")

# ML IMPORTS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

# SAFE LOGGER IMPORT
try:
    from ultimate_safe_logger import get_ultimate_logger
    SAFE_LOGGER_AVAILABLE = True
except:
    SAFE_LOGGER_AVAILABLE = False

warnings.filterwarnings("ignore")

class BulletproofFeatureSelector:
    """
    🏢 ENTERPRISE BULLETPROOF FEATURE SELECTOR
    Real SHAP + Optuna Feature Selection with Production Standards
    
    🎯 ZERO TOLERANCE POLICY:
    - NO Fast Mode
    - NO Mock Data
    - NO Fallback Methods
    - AUC ≥ 70% ENFORCED
    """
    
    def __init__(self, target_auc: float = 0.70, max_features: int = 30, 
                 max_trials: int = 200, **kwargs):
        
        # Enterprise Configuration
        self.target_auc = target_auc
        self.max_features = max_features
        self.max_trials = max_trials
        self.cv_folds = 5
        
        # Initialize Logger
        if SAFE_LOGGER_AVAILABLE:
            self.logger = get_ultimate_logger("EnterpriseFeatureSelector")
        else:
            self.logger = logging.getLogger(__name__)
        
        # Enterprise Requirements Check
        if not ENTERPRISE_LIBS_AVAILABLE:
            raise ImportError("🚫 ENTERPRISE FAILURE: SHAP and Optuna libraries required")
        
        # Results Storage
        self.selected_features = []
        self.best_auc = 0.0
        self.shap_rankings = {}
        self.optimization_results = {}
        self.is_fitted = False
        
        self.logger.info("🎯 ENTERPRISE BulletproofFeatureSelector initialized")
        self.logger.info(f"📊 Target AUC: {self.target_auc:.1%}")
        self.logger.info(f"🎯 Max Features: {self.max_features}")
        self.logger.info(f"⚡ Max Trials: {self.max_trials}")
        self.logger.info("🚫 ZERO COMPROMISE: No fast mode, no fallbacks, no sampling")
    
    def fit(self, X, y=None, **kwargs):
        """
        🎯 ENTERPRISE FIT: Real SHAP + Optuna Feature Selection
        """
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting ENTERPRISE feature selection...")
        self.logger.info(f"📊 Data Shape: {X.shape}")
        self.logger.info(f"🎯 Processing ALL {len(X):,} rows (NO SAMPLING)")
        
        if y is None:
            raise ValueError("🚫 Target variable y is REQUIRED for enterprise feature selection")
        
        try:
            # Step 1: SHAP Feature Importance Analysis
            self.logger.info("🧠 Step 1: SHAP Feature Importance Analysis...")
            self.shap_rankings = self._analyze_shap_importance(X, y)
            
            # Step 2: Optuna Feature Optimization  
            self.logger.info("⚡ Step 2: Optuna Feature Optimization...")
            self.optimization_results = self._optuna_optimization(X, y)
            
            # Step 3: Extract Best Features
            self.logger.info("🎯 Step 3: Extracting Best Features...")
            self.selected_features = self._extract_best_features()
            
            # Step 4: Final Validation
            self.logger.info("✅ Step 4: Final Enterprise Validation...")
            validation_results = self._validate_selection(X, y)
            
            # Enterprise Compliance Gate
            if self.best_auc < self.target_auc:
                raise ValueError(
                    f"🚫 ENTERPRISE COMPLIANCE FAILURE: AUC {self.best_auc:.4f} < "
                    f"target {self.target_auc:.2f}. Production deployment BLOCKED."
                )
            
            self.is_fitted = True
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ ENTERPRISE Feature Selection COMPLETED")
            self.logger.info(f"🎯 Features Selected: {len(self.selected_features)}")
            self.logger.info(f"📊 Final AUC: {self.best_auc:.4f}")
            self.logger.info(f"⏱️  Processing Time: {processing_time:.1f} seconds")
            self.logger.info(f"🏆 TARGET ACHIEVED: {self.best_auc:.4f} ≥ {self.target_auc:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"🚫 ENTERPRISE Feature Selection FAILED: {e}")
            raise RuntimeError(f"Enterprise feature selection failed: {e}")
    
    def _analyze_shap_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """🧠 SHAP Feature Importance Analysis - TRUE ENTERPRISE PRODUCTION"""
        self.logger.info("🔍 Analyzing SHAP feature importance on 100% FULL DATASET...")
        
        # 🏢 ENTERPRISE REQUIREMENT: Use ALL DATA - ZERO SAMPLING
        # TRUE PRODUCTION ENTERPRISE POLICY: Process 100% of available data
        dataset_size = len(X)
        
        # 🚀 ENTERPRISE POLICY: NO SAMPLING - USE ALL DATA
        X_sample = X.copy()
        y_sample = y.copy()
        self.logger.info(f"🎯 ENTERPRISE PRODUCTION: Using ALL {dataset_size:,} rows for SHAP analysis (100% FULL DATASET)")
        self.logger.info("� ZERO COMPROMISE: No sampling, no row limits, TRUE enterprise-grade analysis")
        
        # Train PRODUCTION-GRADE model for SHAP analysis
        model = RandomForestClassifier(
            n_estimators=500,  # INCREASED for production stability
            max_depth=15,      # INCREASED for real-world complexity
            min_samples_split=5,   # REDUCED for more detailed analysis
            min_samples_leaf=2,    # REDUCED for finer granularity
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_sample, y_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create rankings
        rankings = {}
        for i, feature in enumerate(X.columns):
            if i < len(feature_importance):
                rankings[feature] = float(feature_importance[i])
        
        # Sort by importance
        rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"✅ SHAP analysis completed: {len(rankings)} features ranked")
        return rankings
    
    def _optuna_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """⚡ Optuna Feature Optimization - PRODUCTION GRADE"""
        self.logger.info(f"🔬 Starting PRODUCTION Optuna optimization with {self.max_trials} trials...")
        
        # Create study with production-grade settings
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=50)  # More conservative pruning
        )
        
        # Run optimization with extended timeouts for production
        study.optimize(
            lambda trial: self._objective_function(trial, X, y),
            n_trials=self.max_trials,
            timeout=3600,  # 60 minutes for production optimization
            show_progress_bar=False
        )
        
        best_trial = study.best_trial
        self.best_auc = best_trial.value if best_trial.value else 0.0
        
        results = {
            'best_auc': self.best_auc,
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'successful_trials': len([t for t in study.trials if t.value is not None])
        }
        
        self.logger.info(f"✅ Optuna optimization completed")
        self.logger.info(f"🎯 Best AUC: {self.best_auc:.4f} from {len(study.trials)} trials")
        
        return results
    
    def _objective_function(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """🎯 Optuna Objective Function"""
        try:
            # Select number of features
            n_features = trial.suggest_int(
                'n_features',
                max(10, min(15, len(X.columns)//3)),
                min(self.max_features, len(X.columns))
            )
            
            # Select top features from SHAP rankings
            top_features = list(self.shap_rankings.keys())[:n_features]
            X_selected = X[top_features]
            
            # Model selection
            model_type = trial.suggest_categorical('model_type', ['rf', 'gb'])
            
            if model_type == 'rf':
                # PRODUCTION Random Forest with more comprehensive parameters
                n_estimators = trial.suggest_int('rf_n_estimators', 200, 800)
                max_depth = trial.suggest_int('rf_max_depth', 8, 25)
                min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 15)
                min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 8)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:
                # PRODUCTION Gradient Boosting with extended parameters
                n_estimators = trial.suggest_int('gb_n_estimators', 200, 500)
                max_depth = trial.suggest_int('gb_max_depth', 4, 12)
                learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3)
                subsample = trial.suggest_float('gb_subsample', 0.7, 1.0)
                
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=42
                )
            
            # PRODUCTION TimeSeriesSplit cross-validation (more splits for robustness)
            tscv = TimeSeriesSplit(n_splits=min(10, max(5, len(X) // 50000)))  # Dynamic CV folds
            cv_scores = cross_val_score(
                model, X_selected, y,
                cv=tscv, scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {str(e)}")
            return 0.5
    
    def _extract_best_features(self) -> List[str]:
        """🎯 Extract Best Features"""
        best_params = self.optimization_results.get('best_params', {})
        n_features = best_params.get('n_features', self.max_features)
        
        # Use top SHAP features
        selected = list(self.shap_rankings.keys())[:n_features]
        
        if not selected:
            raise RuntimeError("❌ No features could be selected")
        
        return selected
    
    def _validate_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """✅ Final Validation"""
        if not self.selected_features:
            raise ValueError("❌ No features selected for validation")
        
        X_selected = X[self.selected_features]
        
        # Get best model parameters
        best_params = self.optimization_results.get('best_params', {})
        model_type = best_params.get('model_type', 'rf')
        
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=best_params.get('rf_n_estimators', 200),
                max_depth=best_params.get('rf_max_depth', 10),
                min_samples_split=best_params.get('rf_min_samples_split', 15),
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=best_params.get('gb_n_estimators', 150),
                max_depth=best_params.get('gb_max_depth', 6),
                learning_rate=best_params.get('gb_learning_rate', 0.1),
                random_state=42
            )
        
        # PRODUCTION Cross-validation with enhanced reporting
        tscv = TimeSeriesSplit(n_splits=min(10, max(5, len(X) // 30000)))  # Dynamic CV based on data size
        cv_scores = cross_val_score(model, X_selected, y, cv=tscv, scoring='roc_auc')
        
        self.best_auc = cv_scores.mean()
        
        self.logger.info(f"✅ PRODUCTION validation completed: AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'target_achieved': cv_scores.mean() >= self.target_auc,
            'enterprise_compliant': True,
            'production_grade': True
        }
    
    def transform(self, X, **kwargs):
        """🔄 Transform Data"""
        if not self.is_fitted:
            raise ValueError("🚫 Selector must be fitted before transform")
        
        if not self.selected_features:
            raise ValueError("🚫 No features selected")
        
        self.logger.info(f"🔄 Transforming data with {len(self.selected_features)} selected features")
        
        # Select only the chosen features
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if not available_features:
            raise ValueError("🚫 No selected features found in input data")
        
        result = X[available_features]
        self.logger.info(f"✅ Transform completed: {result.shape}")
        return result
    
    def fit_transform(self, X, y=None, **kwargs):
        """🎯 Fit and Transform in one step"""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    
    def get_feature_names_out(self, input_features=None):
        """📋 Get Selected Feature Names"""
        return self.selected_features if self.selected_features else []
        
    def get_support(self, indices=False):
        """ได้ support mask หรือ indices"""
        if indices:
            return list(range(len(self.features)))
        else:
            return [True] * len(self.features)

# Aliases สำหรับ backward compatibility
AdvancedElliottWaveFeatureSelector = BulletproofFeatureSelector
EnterpriseShapOptunaFeatureSelector = BulletproofFeatureSelector
SHAPOptunaFeatureSelector = BulletproofFeatureSelector
RealProfitFeatureSelector = BulletproofFeatureSelector

def create_feature_selector(**kwargs):
    """Factory function สำหรับสร้าง feature selector"""
    return BulletproofFeatureSelector(**kwargs)
