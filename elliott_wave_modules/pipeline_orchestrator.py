#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELLIOTT WAVE PIPELINE ORCHESTRATOR
Elliott Wave Pipeline Controller - Complete System Management

Enterprise Features:
- Complete Pipeline Orchestration
- Component Integration
- Quality Gates Enforcement
- Enterprise Compliance Validation
- Production-Ready Execution
"""

# 🛠️ CUDA FIX: Force CPU-only operation to prevent CUDA errors
import os
import warnings

# Environment variables to force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress CUDA warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Standard library imports
import time
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# 🚀 Import Advanced Logging System
try:
    from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    print("⚠️ Advanced logging not available, using standard logging")

# Import Pipeline Data Container and Enterprise Base
from core.pipeline_data_container import PipelineDataContainer, create_pipeline_container, safe_extract_data
from core.enterprise_component_base import EnterpriseComponentBase

import json

class ElliottWavePipelineOrchestrator(EnterpriseComponentBase):
    """ตัวควบคุม Pipeline Elliott Wave ระดับ Enterprise"""
    
    def __init__(self, data_processor=None, cnn_lstm_engine=None, dqn_agent=None, feature_selector=None, 
                 performance_analyzer=None, logger=None, beautiful_logger=None, output_manager=None,
                 ml_protection=None, resource_manager=None, config=None):
        
        # Initialize enterprise component base
        super().__init__("ElliottWavePipelineOrchestrator", config)
        
        # Component assignments - can be None for testing
        self.data_processor = data_processor
        self.cnn_lstm_engine = cnn_lstm_engine
        self.dqn_agent = dqn_agent
        self.feature_selector = feature_selector
        self.performance_analyzer = performance_analyzer
        self.logger = logger
        self.beautiful_logger = beautiful_logger
        self.output_manager = output_manager
        self.ml_protection = ml_protection
        self.resource_manager = resource_manager
        
        # Pipeline state
        self.pipeline_results = {}
        self.pipeline_container = None  # Will hold PipelineDataContainer
        
        self.log_success("Pipeline orchestrator initialized successfully")

    def run_full_pipeline(self, initial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes the entire pipeline using a single, pre-loaded data source.
        Uses PipelineDataContainer for standardized data flow.
        """
        if initial_data is None or initial_data.empty:
            self.log_error("Orchestrator received no initial data. Aborting pipeline.")
            self.beautiful_logger.fail_step(1, "Pipeline Start", "No data provided to orchestrator.", "❌")
            return {"status": "failed", "error": "No initial data"}

        # Create pipeline data container
        self.pipeline_container = create_pipeline_container(
            data=initial_data,
            pipeline_id=f"elliott_wave_{self.session_id}",
            session_id=self.session_id
        )
        
        self.log_success(f"Pipeline starting with {len(initial_data)} data rows")

        # Define pipeline stages with standardized data flow
        stages = [
            (2, "Data Preprocessing & Feature Engineering", self._stage_2_preprocess_and_feature_engineer),
            (3, "Advanced Feature Selection", self._stage_3_feature_selection),
            (4, "CNN-LSTM Model Training", self._stage_4_cnn_lstm_training),
            (5, "DQN Agent Training", self._stage_5_dqn_training),
            (6, "Performance Analysis", self._stage_6_performance_analysis),
            (7, "Generate Final Report", self._stage_7_generate_report)
        ]

        try:
            for step, name, func in stages:
                self.beautiful_logger.start_step(step, name, f"Executing step: {name}")
                
                # Execute stage with timing
                start_time = time.time()
                result = func() 
                execution_time = time.time() - start_time
                
                if not result.get("success", False):
                    error_msg = result.get('error', 'Unknown error')
                    self.log_error(f"Pipeline failed at step '{name}': {error_msg}")
                    self.beautiful_logger.fail_step(step, name, f"Failed: {error_msg}", "❌")
                    
                    # Add error to pipeline container
                    self.pipeline_container.add_error(error_msg, name)
                    
                    # Log traceback if available
                    if 'traceback' in result:
                        self.logger.debug(result['traceback'])
                    return {"status": "failed", "error": f"Failed at {name}: {error_msg}"}
                
                # Add successful step to pipeline container
                self.pipeline_container.add_step_result(
                    step_name=name,
                    result_data=self.pipeline_container.data,  # Current data state
                    step_metadata=result.get('metadata', {}),
                    execution_time=execution_time,
                    step_status='success'
                )
                
                self.pipeline_results[name] = result
                self.beautiful_logger.complete_step(step, name, result.get('message', 'Completed successfully.'), "✅")

            self.log_success("Pipeline completed all stages successfully")
            return {
                "status": "success", 
                "results": self.pipeline_results,
                "container_summary": self.pipeline_container.to_dict()
            }

        except Exception as e:
            self.log_error(f"Critical unhandled error occurred in the orchestrator", e)
            self.beautiful_logger.fail_step(99, "Critical Failure", f"Unhandled exception: {e}", "🔥")
            self.pipeline_container.add_error(str(e), "pipeline_orchestrator", e)
            return {"status": "failed", "error": str(e)}

    def _stage_2_preprocess_and_feature_engineer(self) -> Dict[str, Any]:
        """
        Processes data for Elliott Wave and adds technical indicators.
        Uses standardized data flow with PipelineDataContainer.
        """
        try:
            self.log_success("Starting Stage 2: Data Preprocessing & Feature Engineering")
            
            # Extract data from pipeline container
            current_data = safe_extract_data(self.pipeline_container)
            
            if current_data is None or current_data.empty:
                return {"success": False, "error": "No data available for preprocessing"}
            
            # Process data with the updated method signature
            processed_data = self.data_processor.process_data_for_elliott_wave(current_data)
            
            if processed_data is None or processed_data.empty:
                return {"success": False, "error": "Data processing returned no data."}

            # Update pipeline container with processed data
            self.pipeline_container.data = processed_data
            
            message = f"Preprocessing complete. Data shape: {processed_data.shape}"
            self.log_success(message)
            
            return {
                "success": True, 
                "message": message, 
                "data_shape": processed_data.shape,
                "metadata": {
                    "processing_method": "elliott_wave_preprocessing",
                    "input_rows": len(current_data),
                    "output_rows": len(processed_data)
                }
            }

        except Exception as e:
            self.log_error(f"Stage 2 preprocessing failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _stage_3_feature_selection(self) -> Dict[str, Any]:
        """
        Selects the best features for the models using standardized data flow.
        """
        try:
            self.log_success("Starting Stage 3: Advanced Feature Selection")
            
            # Extract data from pipeline container
            current_data = safe_extract_data(self.pipeline_container)
            
            if current_data is None or current_data.empty:
                return {"success": False, "error": "No data available for feature selection"}
            
            # Create target variable for feature selection
            data_with_target = current_data.copy()
            data_with_target['target'] = (data_with_target['close'].shift(-1) > data_with_target['close']).astype(int)
            data_with_target.dropna(inplace=True)

            X = data_with_target.drop('target', axis=1)
            y = data_with_target['target']

            # Perform feature selection
            selected_features, prepared_data = self.feature_selector.select_features(X, y)

            if not selected_features or (hasattr(selected_features, '__len__') and len(selected_features) == 0):
                return {"success": False, "error": "Feature selection returned no features."}

            # Update pipeline container with selected features data
            self.pipeline_container.data = prepared_data

            message = f"Selected {len(selected_features)} features from {len(X.columns)} original features"
            self.log_success(message)
            
            return {
                "success": True, 
                "message": message, 
                "selected_features": selected_features,
                "metadata": {
                    "original_features": len(X.columns),
                    "selected_features": len(selected_features),
                    "feature_names": selected_features
                }
            }

        except Exception as e:
            self.log_error(f"Stage 3 feature selection failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _stage_4_cnn_lstm_training(self) -> Dict[str, Any]:
        """
        Trains the CNN-LSTM model using standardized data flow.
        """
        try:
            self.log_success("Starting Stage 4: CNN-LSTM Model Training")
            
            # Extract data from pipeline container
            current_data = safe_extract_data(self.pipeline_container)
            
            # Handle both DataFrame and numpy array inputs
            if current_data is None:
                return {"success": False, "error": "No data available for CNN-LSTM training"}
            
            # Check if data is empty (works for both DataFrame and numpy array)
            if hasattr(current_data, 'empty'):
                if current_data.empty:
                    return {"success": False, "error": "No data available for CNN-LSTM training"}
            elif hasattr(current_data, 'shape'):
                if current_data.shape[0] == 0:
                    return {"success": False, "error": "No data available for CNN-LSTM training"}
            elif len(current_data) == 0:
                return {"success": False, "error": "No data available for CNN-LSTM training"}
            
            # Prepare training data
            X = current_data.drop('target', axis=1)
            y = current_data['target']

            # Train CNN-LSTM model
            model, history = self.cnn_lstm_engine.train_model(X, y)
            
            if model is None:
                return {"success": False, "error": "CNN-LSTM model training failed."}

            # Save the model
            model_path = self.output_manager.save_model(model, 'cnn_lstm_elliott_wave')
            
            message = f"CNN-LSTM model trained successfully. Model saved to {model_path}"
            self.log_success(message)
            
            return {
                "success": True, 
                "message": message, 
                "model_path": str(model_path),
                "metadata": {
                    "training_samples": len(X),
                    "features": len(X.columns),
                    "model_type": "CNN-LSTM"
                }
            }

        except Exception as e:
            self.log_error(f"Stage 4 CNN-LSTM training failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _stage_5_dqn_training(self) -> Dict[str, Any]:
        """
        Trains the DQN agent using standardized data flow.
        """
        try:
            self.log_success("Starting Stage 5: DQN Agent Training")
            
            # Extract data from pipeline container
            current_data = safe_extract_data(self.pipeline_container)
            
            if current_data is None or current_data.empty:
                return {"success": False, "error": "No data available for DQN training"}
            
            # Prepare training data for DQN
            X = current_data.drop('target', axis=1)
            y = current_data['target']

            # Train DQN agent
            training_history = self.dqn_agent.train_agent(X, y)
            
            if training_history is None:
                return {"success": False, "error": "DQN agent training failed."}

            # Save the trained agent
            agent_path = self.output_manager.save_model(self.dqn_agent, 'dqn_elliott_wave_agent')
            
            message = f"DQN agent trained successfully. Agent saved to {agent_path}"
            self.log_success(message)
            
            return {
                "success": True, 
                "message": message, 
                "agent_path": str(agent_path),
                "metadata": {
                    "training_samples": len(X),
                    "features": len(X.columns),
                    "model_type": "DQN",
                    "final_reward": training_history.get('final_reward', 0)
                }
            }

        except Exception as e:
            self.log_error(f"Stage 5 DQN training failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _stage_6_performance_analysis(self) -> Dict[str, Any]:
        """
        Analyzes the performance of the trained models using standardized data flow.
        """
        try:
            self.log_success("Starting Stage 6: Performance Analysis")
            
            # Compile all pipeline results for analysis
            analysis_results = self.performance_analyzer.analyze_performance(self.pipeline_results)
            
            if not analysis_results:
                return {"success": False, "error": "Performance analysis failed."}

            message = f"Performance analysis completed successfully"
            self.log_success(message)
            
            return {
                "success": True,
                "message": message,
                "analysis_results": analysis_results,
                "metadata": {
                    "analysis_type": "comprehensive_performance",
                    "stages_analyzed": len(self.pipeline_results)
                }
            }

        except Exception as e:
            self.log_error(f"Stage 6 performance analysis failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _stage_7_generate_report(self) -> Dict[str, Any]:
        """
        Generates the final summary report of the entire pipeline run using standardized data flow.
        """
        try:
            self.log_success("Starting Stage 7: Generate Final Report")
            
            # Compile comprehensive pipeline report
            pipeline_report = {
                "session_info": {
                    "session_id": self.session_id,
                    "pipeline_id": self.pipeline_container.metadata.get("pipeline_id"),
                    "execution_timestamp": datetime.now().isoformat()
                },
                "pipeline_results": self.pipeline_results,
                "container_summary": self.pipeline_container.to_dict(),
                "performance_summary": self.pipeline_container.get_performance_summary(),
                "status_summary": self.pipeline_container.get_status_summary(),
                "data_flow_summary": self.pipeline_container.get_data_summary()
            }
            
            # Save comprehensive report
            report_path = self.output_manager.save_report(pipeline_report, 'elliott_wave_pipeline_report')

            message = f"Pipeline report generated successfully. Report saved to {report_path}"
            self.log_success(message)
            
            return {
                "success": True,
                "message": message,
                "report_path": str(report_path),
                "report_summary": pipeline_report,
                "metadata": {
                    "report_type": "comprehensive_pipeline_report",
                    "total_stages": len(self.pipeline_results),
                    "execution_time": self.pipeline_container.get_performance_summary().get("total_execution_time", 0)
                }
            }

        except Exception as e:
            self.log_error(f"Stage 7 report generation failed", e)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
            self.logger.info("Starting Stage 7: Generating Final Report...")
            
            final_report = {
                "run_timestamp": datetime.now().isoformat(),
                "pipeline_summary": {
                    "total_stages_completed": len(self.pipeline_results),
                    "initial_data_shape": self.pipeline_results.get("Data Preprocessing & Feature Engineering", {}).get("data_shape"),
                    "selected_features_count": len(self.pipeline_results.get("Advanced Feature Selection", {}).get("selected_features", [])),
                },
                "model_paths": {
                    "cnn_lstm": self.pipeline_results.get("CNN-LSTM Model Training", {}).get("model_path"),
                    "dqn_agent": self.pipeline_results.get("DQN Agent Training", {}).get("agent_path"),
                },
                "performance": self.pipeline_results.get("Performance Analysis", {}).get("report", {}),
                "full_results": self.pipeline_results
            }

            report_path = self.output_manager.save_report(final_report, "NICEGOLD_PROJECTP_MENU1_FINAL_REPORT")
            
            message = f"Final pipeline report generated and saved to {report_path}."
            self.logger.info(message)
            return {"success": True, "message": message, "report_path": str(report_path)}

        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
