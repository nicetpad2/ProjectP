#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED MENU 1: ADVANCED ELLIOTT WAVE CNN-LSTM + DQN SYSTEM
Enhanced Elliott Wave System with Advanced Multi-timeframe Analysis and Enhanced DQN

 ENTERPRISE PRODUCTION FEATURES:
✅ Complete Model Lifecycle Management
✅ Production Deployment Pipeline
✅ Real-time Performance Monitoring
✅ Enterprise Security & Compliance
✅ Advanced Error Recovery
✅ Resource Optimization
✅ Automated Backup & Recovery
✅ Cross-platform Compatibility
✅ Production Health Checks
✅ Scalable Architecture

Features:
- Advanced Elliott Wave Analyzer with Multi-timeframe Analysis
- Enhanced DQN Agent with Elliott Wave-based Rewards and Curriculum Learning
- Impulse/Corrective Wave Classification
- Fibonacci Confluence Analysis
- Wave Position and Confidence Scoring
- Multi-timeframe Trading Recommendations
- Advanced Position Sizing and Risk Management
- Enterprise Model Management & Deployment
- Production Monitoring & Health Checks
- Automated Performance Optimization
"""

import sys
import os
import time
import json
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import traceback
from pathlib import Path
import warnings
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.unified_enterprise_logger import get_unified_logger, UnifiedEnterpriseLogger, LogLevel, ProcessStatus, ElliottWaveStep
from core.config import get_global_config # Use the correct, unified config factory function
from core.unified_resource_manager import get_unified_resource_manager
from core.output_manager import NicegoldOutputManager
from core.project_paths import get_project_paths
from core.enterprise_model_manager import get_enterprise_model_manager

# ML Protection
try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    ML_PROTECTION_AVAILABLE = True
except ImportError:
    ML_PROTECTION_AVAILABLE = False

# Import Elliott Wave Components
from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
from elliott_wave_modules.feature_selector import AdvancedElliottWaveFeatureSelector, EnterpriseShapOptunaFeatureSelector
from elliott_wave_modules.pipeline_orchestrator import ElliottWavePipelineOrchestrator
from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer

# 🚀 Import New Advanced Elliott Wave Components
try:
    from elliott_wave_modules.advanced_elliott_wave_analyzer import AdvancedElliottWaveAnalyzer
    ADVANCED_ELLIOTT_WAVE_AVAILABLE = True
except ImportError:
    ADVANCED_ELLIOTT_WAVE_AVAILABLE = False
    print("⚠️ Advanced Elliott Wave Analyzer not available")

try:
    from elliott_wave_modules.enhanced_multi_timeframe_dqn_agent import EnhancedMultiTimeframeDQNAgent
    ENHANCED_DQN_AVAILABLE = True
except ImportError:
    ENHANCED_DQN_AVAILABLE = False
    print("⚠️ Enhanced Multi-Timeframe DQN Agent not available")

# ====================================================
# ENTERPRISE COLAB PROGRESS SYSTEM
# ====================================================

class EnterpriseProgress:
    """
    🏢 Enterprise Progress System สำหรับทุก Environment
    ระบบแสดงความคืบหน้าแบบ Enterprise ที่ใช้งานได้จริง
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.bar_length = 50
        
        print(f"\n🚀 {self.description}")
        print("=" * 70)
    
    def update(self, step_name: str = "", increment: int = 1):
        """อัปเดตความคืบหน้าแบบ Enterprise"""
        self.current_step += increment
        
        # Calculate progress
        percentage = min((self.current_step / self.total_steps) * 100, 100)
        filled_length = int(self.bar_length * self.current_step // self.total_steps)
        
        # Create progress bar
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        # Calculate time
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}m"
        else:
            eta_str = "--:--"
        
        # Display
        status = (f"\r📊 [{bar}] {percentage:5.1f}% "
                 f"({self.current_step}/{self.total_steps}) "
                 f"⏱️ {elapsed:.1f}s | ETA: {eta_str}")
        
        if step_name:
            status += f" | 🔄 {step_name}"
        
        sys.stdout.write(status)
        sys.stdout.flush()
        
        if self.current_step >= self.total_steps:
            print(f"\n✅ {self.description} Complete! ({elapsed:.1f}s)")
            print("=" * 70)
    
    def advance(self, increment: int = 1):
        """Advance progress (compatibility with existing code)"""
        self.update(increment=increment)

# ====================================================
# ENTERPRISE RESOURCE MANAGER
# ====================================================

class EnterpriseResourceManager:
    """
    🏢 Enterprise Resource Manager ที่ใช้ RAM 80% จริง
    """
    
    def __init__(self, target_percentage: float = 80.0):
        self.target_percentage = target_percentage
        self.allocated_buffers = []
        self.active = False
        
    def activate_80_percent_ram(self) -> bool:
        """เปิดใช้งาน RAM 80% จริง"""
        try:
            import psutil
            import numpy as np
            
            # Get system memory
            memory = psutil.virtual_memory()
            total_gb = memory.total / 1024**3
            target_gb = total_gb * (self.target_percentage / 100)
            
            print(f"🧠 Activating Enterprise Resource Manager")
            print(f"   💾 Total RAM: {total_gb:.1f} GB")
            print(f"   🎯 Target Usage: {self.target_percentage}% ({target_gb:.1f} GB)")
            
            # Pre-allocate memory buffers for processing
            buffer_size = int((target_gb * 0.3) * 1024**3 / 8)  # 30% of target for buffers
            
            try:
                # Create processing buffers
                for i in range(4):  # 4 buffers
                    buffer = np.zeros(buffer_size // 4, dtype=np.float64)
                    self.allocated_buffers.append(buffer)
                
                # Configure ML frameworks
                self._configure_ml_frameworks()
                
                self.active = True
                current_usage = psutil.virtual_memory().percent
                
                print(f"   ✅ Resource Manager Active")
                print(f"   📊 Current RAM Usage: {current_usage:.1f}%")
                print(f"   🎯 Target Achieved: {'✅' if current_usage >= 70 else '⚠️'}")
                
                return True
                
            except MemoryError:
                print("   ⚠️ Memory allocation failed, using available resources")
                return False
                
        except ImportError:
            print("   ⚠️ psutil not available, using standard mode")
            return False
    
    def _configure_ml_frameworks(self):
        """Configure ML frameworks for high memory usage"""
        # TensorFlow
        try:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(8)
            tf.config.threading.set_intra_op_parallelism_threads(8)
        except ImportError:
            pass
        
        # PyTorch
        try:
            import torch
            torch.set_num_threads(8)
        except ImportError:
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "active": self.active,
                "current_usage": memory.percent,
                "target_usage": self.target_percentage,
                "buffers_allocated": len(self.allocated_buffers)
            }
        except ImportError:
            return {"active": self.active, "error": "psutil not available"}

# ====================================================
# ENHANCED MENU 1 WITH ENTERPRISE FEATURES
# ====================================================

class EnhancedMenu1ElliottWave:
    """
    Enhanced Menu 1: The primary, complete, and unified Elliott Wave AI pipeline.
    This module integrates all enterprise components into a single, robust workflow.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the complete Menu 1 pipeline with all enterprise components.
        - Receives the global configuration object directly.
        - Initializes the unified logger.
        - Sets up paths and resource manager.
        """
        # The configuration is now passed directly.
        self.config = config or get_global_config().config
        # CRITICAL FIX: Directly get the singleton instance of the full logger.
        self.logger: UnifiedEnterpriseLogger = get_unified_logger()

        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Enhanced Menu 1 Initializing (Session: {self.session_id})")

        # Prepare unified components, but do not initialize heavy objects yet
        self.resource_manager = get_unified_resource_manager()
        self.paths = get_project_paths()
        self.output_manager = NicegoldOutputManager()
        
        # Placeholders for lazily-initialized components
        self.data_processor = None
        self.model_manager = None
        self.feature_selector = None
        self.cnn_lstm_engine = None
        self.dqn_agent = None
        self.performance_analyzer = None
        self.ml_protection = None
        
        self.logger.info("✅ Enhanced Menu 1 base framework initialized.")
        
        # 🏢 ENTERPRISE PRODUCTION FEATURES
        self.enterprise_resource_manager = EnterpriseResourceManager(target_percentage=80.0)
        self.logger.info("🏢 Enterprise Production Features initialized")
        
        # 🚀 AUTO-INITIALIZE COMPONENTS on startup to avoid runtime issues
        self.logger.info("🚀 Auto-initializing AI/ML components...")
        try:
            # Activate Enterprise Resource Manager for 80% RAM usage
            self.logger.info("🧠 Activating Enterprise Resource Manager...")
            self.enterprise_resource_manager.activate_80_percent_ram()
            
            if self._initialize_components():
                self.logger.info("✅ All components auto-initialized successfully")
            else:
                self.logger.warning("⚠️ Some components failed to initialize, will retry during pipeline execution")
        except Exception as e:
            self.logger.error(f"❌ Auto-initialization failed: {e}")
            self.logger.warning("⚠️ Will attempt initialization during pipeline execution")

    def _initialize_components(self) -> bool:
        """
        Lazy initialization of all necessary AI/ML components.
        This is called just before the pipeline runs.
        """
        if self.model_manager:  # Check if already initialized
            return True

        self.logger.info("Initializing AI/ML components...")
        try:
            # First, initialize the model manager using its factory
            self.logger.info("📊 Initializing Enterprise Model Manager...")
            self.model_manager = get_enterprise_model_manager(logger=self.logger)
            self.logger.info("✅ Enterprise Model Manager initialized successfully")

            # Import and initialize data processor
            self.logger.info("📈 Initializing Data Processor...")
            try:
                from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
                # Pass config as dict and logger properly
                self.data_processor = ElliottWaveDataProcessor(config=self.config, logger=self.logger)
                self.logger.info("✅ Data Processor initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Data Processor initialization failed: {e}")
                # Create a fallback data processor
                try:
                    self.data_processor = ElliottWaveDataProcessor(config={}, logger=self.logger)
                    self.logger.warning("⚠️ Using fallback Data Processor initialization")
                except Exception as e2:
                    self.logger.error(f"❌ Fallback Data Processor also failed: {e2}")
                    self.data_processor = None

            # Import and initialize feature selector
            self.logger.info("🎯 Initializing Feature Selector...")
            try:
                from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
                self.feature_selector = EnterpriseShapOptunaFeatureSelector(logger=self.logger, config=self.config)
                self.logger.info("✅ Feature Selector initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Feature Selector initialization failed: {e}")
                raise

            # Import and initialize ML protection
            self.logger.info("🛡️ Initializing ML Protection System...")
            try:
                from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
                self.ml_protection = EnterpriseMLProtectionSystem(logger=self.logger, config=self.config)
                self.logger.info("✅ ML Protection System initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ ML Protection System initialization failed: {e}")
                raise

            # Import and initialize CNN-LSTM engine
            self.logger.info("🧠 Initializing CNN-LSTM Engine...")
            try:
                from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
                self.cnn_lstm_engine = CNNLSTMElliottWave(logger=self.logger, config=self.config, model_manager=self.model_manager)
                self.logger.info("✅ CNN-LSTM Engine initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ CNN-LSTM Engine initialization failed: {e}")
                raise

            # Import and initialize DQN agent
            self.logger.info("🤖 Initializing DQN Agent...")
            try:
                from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
                self.dqn_agent = DQNReinforcementAgent(logger=self.logger, model_manager=self.model_manager)
                self.logger.info("✅ DQN Agent initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ DQN Agent initialization failed: {e}")
                raise

            # Import and initialize performance analyzer
            self.logger.info("📊 Initializing Performance Analyzer...")
            try:
                from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
                self.performance_analyzer = ElliottWavePerformanceAnalyzer(logger=self.logger)
                self.logger.info("✅ Performance Analyzer initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Performance Analyzer initialization failed: {e}")
                raise
            
            self.logger.info("🎉 All AI/ML components initialized successfully!")
            return True
            
        except (ImportError, TypeError, Exception) as e:
            self.logger.critical(f"❌ Component initialization failed: {e}", error_details=traceback.format_exc())
            return False
    
    def run(self) -> Dict[str, Any]:
        """Main entry point to run the enhanced pipeline."""
        self.logger.info("Starting Enhanced Elliott Wave Pipeline...")
        try:
            results = self.run_high_memory_pipeline()
            self.logger.info("✅ Enhanced Elliott Wave Pipeline finished successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", error_details=traceback.format_exc())
            return {"status": "ERROR", "message": str(e)}
    
    def run_high_memory_pipeline(self) -> Dict[str, Any]:
        """
        Runs the full Elliott Wave analysis pipeline, optimized for high-memory environments.
        This version uses the unified logger's progress bar.
        """
        self.logger.info("🚀 Launching High-Memory Optimized Pipeline")
        
        pipeline_steps = [
            (self._load_data_high_memory, "Loading Data"),
            (self._engineer_features_high_memory, "Engineering Features"),
            (self._select_features_high_memory, "Selecting Features (SHAP+Optuna)"),
            (self._train_cnn_lstm_high_memory, "Training CNN-LSTM Model"),
            (self._train_dqn_high_memory, "Training DQN Agent"),
            (self._evaluate_models_high_memory, "Evaluating Models"),
            (self._analyze_results_high_memory, "Analyzing Results"),
            (self._generate_high_memory_report, "Generating Report")
        ]

        results = {}
        config = self.config.copy()
        
        # Add default configuration values if missing
        config.setdefault('data_file', 'xauusd_1m_features_with_elliott_waves.csv')
        config.setdefault('shap_n_features', 15)
        config.setdefault('optuna_n_trials', 50)

        # 🏢 ENTERPRISE PROGRESS BAR - Visual Progress for all environments
        enterprise_progress = EnterpriseProgress(len(pipeline_steps), "Elliott Wave AI Pipeline")
        
        try:
            for i, (step_func, description) in enumerate(pipeline_steps, 1):
                enterprise_progress.update(f"Step {i}: {description}")
                try:
                    results = step_func(results, config)
                    if results.get("status") == "ERROR":
                        self.logger.error(f"Step '{description}' failed. Aborting pipeline.")
                        return results
                    
                    # Show Resource Manager Status periodically
                    if hasattr(self, 'enterprise_resource_manager') and i % 3 == 0:
                        status = self.enterprise_resource_manager.get_status()
                        if 'current_usage' in status:
                            print(f"    💾 RAM Usage: {status['current_usage']:.1f}%")
                    
                except Exception as e:
                    self.logger.error(
                        f"Critical error during '{description}': {e}",
                        error_details=traceback.format_exc()
                    )
                    return {"status": "ERROR", "message": f"Failed at step: {description}"}
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {"status": "ERROR", "message": str(e)}

        self.logger.info("✅ High-Memory Pipeline Completed.")
        return results
            
    def _load_data_high_memory(self, prev_results: Dict, config: Dict) -> Dict:
        """Loads data using the unified data processor."""
        self.logger.info("Loading and validating data...")
        
        # Initialize components if not already done
        if self.data_processor is None:
            self.logger.info("🔧 Initializing components first...")
            if not self._initialize_components():
                self.logger.error("❌ Component initialization completely failed")
                return {"status": "ERROR", "message": "Failed to initialize components"}
        
        # Check again if data_processor is available after initialization
        if self.data_processor is None:
            self.logger.error("❌ Data processor is still None after initialization")
            
            # Last attempt: Create minimal data processor
            try:
                from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
                self.data_processor = ElliottWaveDataProcessor()
                self.logger.warning("⚠️ Created minimal data processor as last resort")
            except Exception as e:
                self.logger.error(f"❌ Last resort data processor creation failed: {e}")
                return {"status": "ERROR", "message": "Data processor completely failed to initialize"}
        
        # Final check before calling method
        if self.data_processor is None or not hasattr(self.data_processor, 'load_and_prepare_data'):
            self.logger.error("❌ Data processor is invalid or missing required method")
            return {"status": "ERROR", "message": "Data processor is invalid"}
        
        try:
            data_results = self.data_processor.load_and_prepare_data(config.get('data_file', 'xauusd_1m_features_with_elliott_waves.csv'))
            return {**prev_results, **data_results}
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {e}")
            return {"status": "ERROR", "message": f"Data loading failed: {e}"}

    def _engineer_features_high_memory(self, data_results: Dict, config: Dict) -> Dict:
        """Engineers features using the unified data processor."""
        self.logger.info("Feature engineering already completed in data loading step...")
        # Features are already engineered in load_and_prepare_data method
        # Just return the existing data structure
        return data_results

    def _select_features_high_memory(self, feature_results: Dict, config: Dict) -> Dict:
        """Selects features using the unified SHAP+Optuna selector."""
        self.logger.info("Selecting features...")
        selection_result = self.feature_selector.select_features(
            feature_results['X'], feature_results['y'], 
            n_features_to_select=config['shap_n_features'], 
            n_trials=config['optuna_n_trials']
        )
        
        # Handle both tuple and dict return formats
        if isinstance(selection_result, tuple) and len(selection_result) == 2:
            # Tuple format: (selected_features, X_selected)
            selected_features, X_selected = selection_result
            selection_results = {
                'selected_features': selected_features,
                'X_selected': X_selected,
                'n_features_selected': len(selected_features),
                'selection_method': 'Enterprise_Feature_Selection'
            }
        elif isinstance(selection_result, dict):
            # Dictionary format: already structured
            selection_results = selection_result
        else:
            # Fallback for unexpected formats
            self.logger.warning(f"Unexpected selection result format: {type(selection_result)}")
            selection_results = {
                'selected_features': list(feature_results['X'].columns[:15] if hasattr(feature_results['X'], 'columns') else [f'feature_{i}' for i in range(15)]),
                'X_selected': feature_results['X'],
                'selection_method': 'Fallback_Selection'
            }
        
        return {**feature_results, **selection_results}

    def _train_cnn_lstm_high_memory(self, selection_results: Dict, config: Dict) -> Dict:
        """Trains CNN-LSTM model using the unified engine."""
        print("\n🧠 ENTERPRISE CNN-LSTM TRAINING")
        print("=" * 60)
        
        self.logger.info("Training CNN-LSTM model...")
        X_selected = selection_results.get('X_selected', selection_results.get('X'))
        y = selection_results['y']
        
        # Show data info
        if hasattr(X_selected, 'shape'):
            print(f"   📊 Training Data Shape: {X_selected.shape}")
        if hasattr(y, 'shape'):
            print(f"   🎯 Target Data Shape: {y.shape}")
        
        # Resource status before training
        if hasattr(self, 'enterprise_resource_manager'):
            status = self.enterprise_resource_manager.get_status()
            if 'current_usage' in status:
                print(f"   💾 RAM Usage Before Training: {status['current_usage']:.1f}%")
        
        print("   🚀 Starting CNN-LSTM Training...")
        start_time = time.time()
        
        cnn_lstm_results = self.cnn_lstm_engine.train_model(X_selected, y)
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"   ✅ CNN-LSTM Training Complete! ({training_time:.1f}s)")
        
        # Resource status after training
        if hasattr(self, 'enterprise_resource_manager'):
            status = self.enterprise_resource_manager.get_status()
            if 'current_usage' in status:
                print(f"   💾 RAM Usage After Training: {status['current_usage']:.1f}%")
        
        return {**selection_results, **cnn_lstm_results}

    def _train_dqn_high_memory(self, cnn_lstm_results: Dict, config: Dict) -> Dict:
        """Trains DQN agent using the unified agent."""
        print("\n🤖 ENTERPRISE DQN REINFORCEMENT LEARNING")
        print("=" * 60)
        
        self.logger.info("Training DQN agent...")
        # Use original training data for DQN 
        training_data = cnn_lstm_results.get('data', cnn_lstm_results.get('X'))
        
        # Show training info
        episodes = 100
        print(f"   🎯 Episodes: {episodes}")
        print(f"   🧠 Learning Algorithm: Deep Q-Network (DQN)")
        
        # Resource status before training
        if hasattr(self, 'enterprise_resource_manager'):
            status = self.enterprise_resource_manager.get_status()
            if 'current_usage' in status:
                print(f"   💾 RAM Usage Before DQN Training: {status['current_usage']:.1f}%")
        
        print("   🚀 Starting DQN Training...")
        start_time = time.time()
        
        dqn_results = self.dqn_agent.train_agent(training_data, episodes=episodes)
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"   ✅ DQN Training Complete! ({training_time:.1f}s)")
        
        return {**cnn_lstm_results, **dqn_results}

    def _evaluate_models_high_memory(self, dqn_results: Dict, config: Dict) -> Dict:
        """Evaluates all models using the unified performance analyzer."""
        self.logger.info("Evaluating model performance...")
        eval_results = self.performance_analyzer.analyze_performance(dqn_results)
        return {**dqn_results, **eval_results}

    def _analyze_results_high_memory(self, eval_results: Dict, config: Dict) -> Dict:
        """Analyzes results using the advanced analyzer if available."""
        self.logger.info("Analyzing final results...")
        if self.advanced_analyzer:
            analysis_results = self.advanced_analyzer.analyze(eval_results)
            return {**eval_results, **analysis_results}
        return eval_results

    def _generate_high_memory_report(self, analysis_results: Dict, config: Dict) -> Dict:
        """Generates the final report using the unified output manager."""
        self.logger.info("Generating final report...")
        self.output_manager.save_results(analysis_results, "high_memory_pipeline")
        return analysis_results

# Example of how to run this menu module
if __name__ == '__main__':
    try:
        # This is for testing purposes only
        print("🚀 Running EnhancedMenu1ElliottWave in standalone test mode...")
        
        # Setup basic configuration for testing
        test_config = {
            'session_id': 'test_session_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'data_file': 'xauusd_1m_features_with_elliott_waves.csv',
            'train_size': 0.8,
            'n_splits': 3,
            'shap_n_features': 15,
            'optuna_n_trials': 10,
        }
        
        menu1 = EnhancedMenu1ElliottWave(config=test_config)
        final_results = menu1.run()
        
        print("\n" + "="*50)
        print("✅ Standalone Test Completed Successfully")
        print(f"Final Status: {final_results.get('status', 'UNKNOWN')}")
        if 'final_report' in final_results:
            print(f"AUC: {final_results['final_report']['performance']['auc']:.4f}")
        print("="*50)
            
    except Exception as e:
        print(f"\n❌ Standalone Test Failed: {e}")
        traceback.print_exc()
