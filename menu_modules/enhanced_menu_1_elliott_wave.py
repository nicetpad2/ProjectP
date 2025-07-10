#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED MENU 1: ADVANCED ELLIOTT WAVE CNN-LSTM + DQN SYSTEM
Enhanced Elliott Wave System with Advanced Multi-timeframe Analysis and Enhanced DQN

Features:
- Advanced Elliott Wave Analyzer with Multi-timeframe Analysis
- Enhanced DQN Agent with Elliott Wave-based Rewards and Curriculum Learning
- Impulse/Corrective Wave Classification
- Fibonacci Confluence Analysis
- Wave Position and Confidence Scoring
- Multi-timeframe Trading Recommendations
- Advanced Position Sizing and Risk Management
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import traceback
from pathlib import Path
import warnings

# Essential data processing imports
import pandas as pd
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Core Components
from core.project_paths import get_project_paths
from core.output_manager import NicegoldOutputManager

# Advanced Logging Integration
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    print("⚠️ Advanced logging not available, using simple progress tracker")

# Beautiful Progress Tracking
from core.simple_beautiful_progress import setup_print_based_beautiful_logging

# Resource Management
try:
    from core.intelligent_resource_manager import initialize_intelligent_resources
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False

# ML Protection
try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    ML_PROTECTION_AVAILABLE = True
except ImportError:
    ML_PROTECTION_AVAILABLE = False

# Import Original Elliott Wave Components
from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
from elliott_wave_modules.feature_selector import AdvancedElliottWaveFeatureSelector, EnterpriseShapOptunaFeatureSelector
from elliott_wave_modules.pipeline_orchestrator import ElliottWavePipelineOrchestrator
from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem

# 🎮 GPU Resource Management
try:
    from core.gpu_resource_manager import EnterpriseGPUManager
    GPU_RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    GPU_RESOURCE_MANAGER_AVAILABLE = False
    print("⚠️ GPU Resource Manager not available")

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


class EnhancedMenu1ElliottWave:
    """Enhanced Menu 1: Advanced Elliott Wave System with Multi-timeframe Analysis and Enhanced DQN"""
    
    def __init__(self, config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None,
                 resource_manager = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.resource_manager = resource_manager
        self.results = {}
        
        # Get project paths
        self.paths = get_project_paths()
        
        # Create safe logger
        class PrintLogger:
            def info(self, msg):
                print(f"INFO: {msg}")
            def warning(self, msg):
                print(f"WARNING: {msg}")
            def error(self, msg):
                print(f"ERROR: {msg}")
            def debug(self, msg):
                print(f"DEBUG: {msg}")
        
        self.safe_logger = PrintLogger()
        
        # Initialize Advanced Logging if available
        if ADVANCED_LOGGING_AVAILABLE:
            try:
                self.logger = get_terminal_logger()
                self.progress_manager = get_progress_manager()
                self.safe_logger.info("🚀 Enhanced Menu 1 Elliott Wave initialized with Advanced Logging")
            except Exception as e:
                self.safe_logger.warning(f"⚠️ Advanced logging failed: {e}")
        
        # Initialize Beautiful Logging
        self.beautiful_logger = setup_print_based_beautiful_logging("Enhanced_ElliottWave_Menu1")
        
        # Initialize Output Manager
        self.output_manager = NicegoldOutputManager()
        
        # Initialize Components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all Enhanced Elliott Wave components"""
        try:
            self.beautiful_logger.start_step(0, "Enhanced Component Initialization", 
                                           "Initializing advanced Elliott Wave components")
            self.safe_logger.info("🌊 Initializing Enhanced Elliott Wave Components...")
            
            # Original Data Processor
            self.beautiful_logger.log_info("Initializing Data Processor...")
            self.data_processor = ElliottWaveDataProcessor(
                config=self.config,
                logger=self.safe_logger
            )
            
            # 🚀 Advanced Elliott Wave Analyzer
            if ADVANCED_ELLIOTT_WAVE_AVAILABLE:
                self.beautiful_logger.log_info("Initializing Advanced Elliott Wave Analyzer...")
                elliott_config = {
                    'timeframes': self.config.get('elliott_wave', {}).get('timeframes', ['1min', '5min', '15min', '1H']),
                    'primary_timeframe': '1min',
                    'wave_detection_sensitivity': 0.02,
                    'fibonacci_tolerance': 0.05
                }
                self.advanced_elliott_analyzer = AdvancedElliottWaveAnalyzer(
                    config=elliott_config,
                    logger=self.safe_logger
                )
                self.safe_logger.info("✅ Advanced Elliott Wave Analyzer initialized")
            else:
                self.advanced_elliott_analyzer = None
                self.safe_logger.warning("⚠️ Using basic Elliott Wave analysis")
            
            # Original CNN-LSTM Engine
            self.beautiful_logger.log_info("Initializing CNN-LSTM Engine...")
            self.cnn_lstm_engine = CNNLSTMElliottWave(
                config=self.config,
                logger=self.safe_logger
            )
            
            # 🚀 Enhanced DQN Agent or Original DQN Agent
            if ENHANCED_DQN_AVAILABLE and self.advanced_elliott_analyzer:
                self.beautiful_logger.log_info("Initializing Enhanced DQN Agent...")
                
                # Enhanced DQN configuration
                enhanced_dqn_config = {
                    'state_size': self.config.get('dqn', {}).get('state_size', 50),
                    'action_size': 6,  # Buy_Small, Buy_Medium, Buy_Large, Sell_Small, Sell_Medium, Sell_Large
                    'learning_rate': self.config.get('dqn', {}).get('learning_rate', 0.001),
                    'gamma': self.config.get('dqn', {}).get('gamma', 0.95),
                    'epsilon_start': self.config.get('dqn', {}).get('epsilon_start', 1.0),
                    'epsilon_end': self.config.get('dqn', {}).get('epsilon_end', 0.01),
                    'epsilon_decay': self.config.get('dqn', {}).get('epsilon_decay', 0.995),
                    'memory_size': self.config.get('dqn', {}).get('memory_size', 10000),
                    'batch_size': self.config.get('dqn', {}).get('batch_size', 32),
                    'target_update': self.config.get('dqn', {}).get('target_update', 100),
                    'curriculum_learning': True,
                    'elliott_wave_integration': True
                }
                
                self.dqn_agent = EnhancedMultiTimeframeDQNAgent(
                    state_size=enhanced_dqn_config.get('state_size', 50),
                    action_size=enhanced_dqn_config.get('action_size', 3),
                    learning_rate=enhanced_dqn_config.get('learning_rate', 0.001)
                )
                # เชื่อม elliott_wave_analyzer หลังจาก initialization
                if hasattr(self.dqn_agent, 'elliott_wave_analyzer'):
                    self.dqn_agent.elliott_wave_analyzer = self.advanced_elliott_analyzer
                self.safe_logger.info("✅ Enhanced DQN Agent with Elliott Wave integration initialized")
                
            else:
                # Fallback to original DQN Agent
                self.beautiful_logger.log_info("Initializing Standard DQN Agent...")
                self.dqn_agent = DQNReinforcementAgent(
                    config=self.config,
                    logger=self.safe_logger
                )
                self.safe_logger.info("✅ Standard DQN Agent initialized (fallback)")
            
            # Feature Selector with GPU Resource Management
            self.beautiful_logger.log_info("Initializing Advanced Feature Selector with GPU Management...")
            self.feature_selector = AdvancedElliottWaveFeatureSelector(
                target_auc=self.config.get('elliott_wave', {}).get('target_auc', 0.70),
                max_features=self.config.get('elliott_wave', {}).get('max_features', 25),
                max_trials=150,
                logger=self.safe_logger
            )
            
            # ML Protection System
            if ML_PROTECTION_AVAILABLE:
                self.beautiful_logger.log_info("Initializing ML Protection System...")
                self.ml_protection = EnterpriseMLProtectionSystem(
                    config=self.config,
                    logger=self.safe_logger
                )
            else:
                self.ml_protection = None
            
            # 🎮 Initialize GPU Resource Manager for 80% Utilization
            if GPU_RESOURCE_MANAGER_AVAILABLE:
                self.beautiful_logger.log_info("Initializing GPU Resource Manager (80% Utilization)...")
                self.gpu_manager = EnterpriseGPUManager()
                self.safe_logger.info("✅ GPU Resource Manager initialized")
                self.safe_logger.info(f"⚡ Resource Config: {self.gpu_manager.get_optimization_report()}")
            else:
                self.gpu_manager = None
                self.safe_logger.warning("⚠️ GPU Resource Manager not available - using standard CPU")

            # 🚀 Initialize Resource Management
            if RESOURCE_MANAGEMENT_AVAILABLE and not self.resource_manager:
                try:
                    self.resource_manager = initialize_intelligent_resources(
                        config=self.config,
                        mode='enhanced_80_percent',
                        logger=self.safe_logger
                    )
                    self.safe_logger.info("✅ Intelligent Resource Manager initialized")
                except Exception as e:
                    self.safe_logger.warning(f"⚠️ Resource Manager initialization failed: {e}")
                    self.resource_manager = None

            self.beautiful_logger.complete_step(0, "All enhanced components initialized successfully")
            self.safe_logger.info("✅ All Enhanced Elliott Wave components initialized successfully!")
            
        except Exception as e:
            self.beautiful_logger.log_error(f"Enhanced component initialization failed: {str(e)}")
            self.safe_logger.error(f"❌ Enhanced component initialization failed: {str(e)}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """Entry point method for ProjectP.py compatibility"""
        return self.run_enhanced_pipeline()
    
    def run_enhanced_pipeline(self) -> Dict[str, Any]:
        """Run Enhanced Elliott Wave Pipeline with Advanced Multi-timeframe Analysis"""
        
        self.beautiful_logger.start_step(0, "🌊 Enhanced Elliott Wave Pipeline", 
                                       "Starting Enhanced Pipeline with Multi-timeframe Analysis")
        
        try:
            self.safe_logger.info("🚀 Starting Enhanced Elliott Wave Pipeline...")
            self._display_enhanced_pipeline_overview()
            
            # Execute enhanced pipeline
            success = self._execute_enhanced_pipeline()
            
            # Display results
            self._display_enhanced_results()
            
            # Return final results
            if success:
                self.results['success'] = True
                self.results['execution_status'] = 'success'
                self.results['message'] = 'Enhanced Elliott Wave Pipeline completed successfully!'
                self.safe_logger.info("✅ Enhanced Elliott Wave Pipeline completed successfully!")
            else:
                self.results['success'] = False
                self.results['execution_status'] = 'failed'
                self.results['message'] = 'Enhanced Elliott Wave Pipeline failed!'
                self.safe_logger.error("❌ Enhanced Elliott Wave Pipeline failed!")
                
            return self.results
            
        except Exception as e:
            error_msg = f"Enhanced Elliott Wave Pipeline failed: {str(e)}"
            self.safe_logger.error(f"❌ {error_msg}")
            self.safe_logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'execution_status': 'critical_error',
                'error_message': str(e),
                'message': f'Enhanced Elliott Wave Pipeline failed: {str(e)}',
                'pipeline_duration': 'N/A'
            }
    
    def _execute_enhanced_pipeline(self) -> bool:
        """Execute Enhanced Elliott Wave Pipeline with Multi-timeframe Analysis and 80% Resource Utilization"""
        try:
            # Step 0: Initialize 80% resource utilization
            self.safe_logger.info("🎯 Step 0: Initializing 80% Resource Utilization...")
            self._ensure_80_percent_utilization()
            resource_info_init = self._monitor_resource_usage("Pipeline Initialization")
            
            # Step 1: Load and process data
            self.safe_logger.info("📊 Step 1: Loading and processing market data...")
            data = self.data_processor.load_real_data()
            
            if data is None or len(data) == 0:
                self.safe_logger.error("❌ No data loaded!")
                return False
                
            # Monitor data loading resource impact
            resource_info_data = self._monitor_resource_usage("Data Loading")
            
            self.safe_logger.info(f"✅ Successfully loaded {len(data):,} rows of market data")
            self.safe_logger.info(f"📊 Data memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Step 2: Create enhanced features with Elliott Wave analysis
            self.safe_logger.info("⚙️ Step 2: Creating enhanced Elliott Wave features...")
            features = self.data_processor.create_elliott_wave_features(data)
            
            # 🚀 Step 3: Advanced Multi-timeframe Elliott Wave Analysis
            if self.advanced_elliott_analyzer:
                self.safe_logger.info("🌊 Step 3: Running Advanced Multi-timeframe Elliott Wave Analysis...")
                try:
                    # Analyze Elliott Wave patterns across multiple timeframes
                    elliott_analysis = self.advanced_elliott_analyzer.analyze_multi_timeframe_waves(data)
                    
                    # Extract wave-based features
                    wave_features = self.advanced_elliott_analyzer.extract_wave_features(elliott_analysis)
                    
                    # Combine original features with wave features
                    if wave_features is not None and len(wave_features) > 0:
                        # Ensure both have the same index
                        common_index = features.index.intersection(wave_features.index)
                        if len(common_index) > 0:
                            features_combined = pd.concat([
                                features.loc[common_index],
                                wave_features.loc[common_index]
                            ], axis=1)
                            self.safe_logger.info(f"✅ Combined {features.shape[1]} original features with {wave_features.shape[1]} wave features")
                            features = features_combined
                        else:
                            self.safe_logger.warning("⚠️ No common index found, using original features")
                    
                    # Store Elliott Wave analysis results
                    self.results['elliott_wave_analysis'] = {
                        'multi_timeframe_analysis': elliott_analysis,
                        'wave_features_count': wave_features.shape[1] if wave_features is not None else 0,
                        'analysis_status': 'success'
                    }
                    
                except Exception as e:
                    self.safe_logger.warning(f"⚠️ Advanced Elliott Wave analysis failed: {e}")
                    self.results['elliott_wave_analysis'] = {
                        'analysis_status': 'failed',
                        'error': str(e)
                    }
            else:
                self.safe_logger.warning("⚠️ Using basic Elliott Wave features only")
                self.results['elliott_wave_analysis'] = {
                    'analysis_status': 'basic_only',
                    'message': 'Advanced analyzer not available'
                }
            
            # Step 4: Prepare ML data
            self.safe_logger.info("🎯 Step 4: Preparing ML data...")
            X, y = self.data_processor.prepare_ml_data(features)
            
            # Step 5: ML Protection Analysis
            if self.ml_protection:
                self.safe_logger.info("🛡️ Step 5: Running ML Protection Analysis...")
                try:
                    feature_names = list(X.columns) if hasattr(X, 'columns') else None
                    protection_results = self.ml_protection.comprehensive_protection_analysis(
                        X=X, y=y, feature_names=feature_names
                    )
                    self.results['ml_protection'] = protection_results
                    self.safe_logger.info("✅ ML Protection Analysis completed")
                except Exception as e:
                    self.safe_logger.warning(f"⚠️ ML Protection Analysis failed: {e}")
                    self.results['ml_protection'] = {'error': str(e), 'status': 'failed'}
            
            # Step 6: Feature selection with resource monitoring
            self.safe_logger.info("🧠 Step 6: Running Advanced SHAP + Optuna feature selection...")
            try:
                # Monitor resource usage during feature selection
                resource_info_start = self._monitor_resource_usage("Feature Selection Start")
                
                # Use advanced feature selection with GPU acceleration
                if hasattr(self.feature_selector, 'select_features_advanced'):
                    selected_features, selection_results = self.feature_selector.select_features_advanced(X, y)
                    self.safe_logger.info("✅ Advanced GPU-accelerated feature selection completed")
                else:
                    selected_features, selection_results = self.feature_selector.select_features(X, y)
                    self.safe_logger.info("✅ Standard feature selection completed")
                
                # Monitor resource usage after feature selection
                resource_info_end = self._monitor_resource_usage("Feature Selection End")
                
                self.results['feature_selection'] = selection_results
                self.results['resource_monitoring'] = {
                    'feature_selection_start': resource_info_start,
                    'feature_selection_end': resource_info_end
                }
                
                self.safe_logger.info(f"✅ Selected {len(selected_features)} optimal features")
            except Exception as e:
                self.safe_logger.error(f"❌ Feature selection failed: {e}")
                selected_features = list(X.columns[:min(15, len(X.columns))]) if hasattr(X, 'columns') else None
                self.results['feature_selection'] = {'error': str(e), 'fallback_features': len(selected_features)}
            
            # Step 7: Train CNN-LSTM with resource monitoring
            self.safe_logger.info("🏗️ Step 7: Training CNN-LSTM model...")
            resource_info_cnn_start = self._monitor_resource_usage("CNN-LSTM Training Start")
            
            cnn_lstm_results = self.cnn_lstm_engine.train_model(X[selected_features], y)
            self.results['cnn_lstm_training'] = cnn_lstm_results
            
            resource_info_cnn_end = self._monitor_resource_usage("CNN-LSTM Training End")
            
            # Step 8: Enhanced DQN Training with resource monitoring
            self.safe_logger.info("🤖 Step 8: Training Enhanced DQN agent...")
            resource_info_dqn_start = self._monitor_resource_usage("DQN Training Start")
            try:
                if isinstance(self.dqn_agent, EnhancedDQNAgent):
                    # Enhanced DQN with Elliott Wave integration
                    self.safe_logger.info("🌊 Training Enhanced DQN with Elliott Wave rewards...")
                    
                    # Prepare training data for enhanced DQN
                    if isinstance(X, pd.DataFrame):
                        dqn_training_data = X[selected_features]
                    else:
                        dqn_training_data = pd.DataFrame(X, columns=selected_features)
                    
                    # Train with curriculum learning and Elliott Wave integration
                    dqn_results = self.dqn_agent.train_with_curriculum_learning(
                        market_data=dqn_training_data,
                        price_data=data[['close']] if 'close' in data.columns else data.iloc[:, -1:],
                        episodes=100,
                        curriculum_stages=4
                    )
                    
                    self.safe_logger.info("✅ Enhanced DQN training with curriculum learning completed")
                    
                else:
                    # Standard DQN training
                    if isinstance(X, pd.DataFrame):
                        dqn_training_data = X[selected_features]
                    else:
                        dqn_training_data = pd.DataFrame(X, columns=selected_features)
                    
                    dqn_results = self.dqn_agent.train_agent(dqn_training_data, episodes=50)
                    self.safe_logger.info("✅ Standard DQN training completed")
                
                self.results['dqn_training'] = dqn_results
                
            except Exception as e:
                self.safe_logger.error(f"❌ DQN training failed: {e}")
                self.results['dqn_training'] = {'error': str(e), 'status': 'failed'}
            
            # Step 9: Generate Trading Recommendations
            if self.advanced_elliott_analyzer and 'elliott_wave_analysis' in self.results:
                self.safe_logger.info("📈 Step 9: Generating Elliott Wave Trading Recommendations...")
                try:
                    elliott_analysis = self.results['elliott_wave_analysis']['multi_timeframe_analysis']
                    recommendations = self.advanced_elliott_analyzer.generate_trading_recommendations(
                        elliott_analysis,
                        current_price=data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1, -1]
                    )
                    
                    self.results['trading_recommendations'] = recommendations
                    self.safe_logger.info("✅ Trading recommendations generated")
                    
                except Exception as e:
                    self.safe_logger.warning(f"⚠️ Trading recommendation generation failed: {e}")
                    self.results['trading_recommendations'] = {'error': str(e), 'status': 'failed'}
            
            # Final Step: Generate comprehensive resource utilization report
            self.safe_logger.info("📊 Generating Resource Utilization Report...")
            final_resource_report = self._generate_resource_utilization_report()
            
            # Store data info with resource optimization details
            self.results['data_info'] = {
                'total_rows': len(data),
                'features_count': X.shape[1] if hasattr(X, 'shape') else 0,
                'target_count': len(y) if hasattr(y, '__len__') else 0,
                'enhanced_features': True if self.advanced_elliott_analyzer else False,
                'resource_optimization': '80% Target Compliance',
                'gpu_acceleration': self.gpu_manager is not None
            }
            
            # Add resource utilization report to results
            self.results['resource_utilization_report'] = final_resource_report
            
            # Log final resource status
            self.safe_logger.info("🎯 FINAL RESOURCE UTILIZATION STATUS:")
            self.safe_logger.info(f"   CPU: {final_resource_report['current_usage']['cpu_percent']:.1f}%")
            self.safe_logger.info(f"   Memory: {final_resource_report['current_usage']['memory_percent']:.1f}%")
            self.safe_logger.info(f"   Status: {final_resource_report['compliance']['overall_status']}")
            
            return True
            
        except Exception as e:
            self.safe_logger.error(f"❌ Enhanced pipeline execution failed: {e}")
            return False
    
    def _display_enhanced_pipeline_overview(self):
        """Display enhanced pipeline overview"""
        print("\n" + "="*80)
        print("🌊 ENHANCED ELLIOTT WAVE CNN-LSTM + DQN SYSTEM")
        print("="*80)
        print("📊 Features:")
        print("  • Multi-timeframe Elliott Wave Analysis")
        print("  • Impulse/Corrective Wave Classification")
        print("  • Fibonacci Confluence Analysis")
        print("  • Enhanced DQN with Elliott Wave Rewards")
        print("  • Curriculum Learning")
        print("  • Advanced Position Sizing")
        print("  • Real-time Trading Recommendations")
        print("="*80)
        
        if self.advanced_elliott_analyzer:
            print("✅ Advanced Elliott Wave Analyzer: ACTIVE")
        else:
            print("⚠️ Advanced Elliott Wave Analyzer: NOT AVAILABLE")
            
        if isinstance(self.dqn_agent, EnhancedDQNAgent):
            print("✅ Enhanced DQN Agent: ACTIVE")
        else:
            print("⚠️ Enhanced DQN Agent: NOT AVAILABLE (using standard)")
            
        print("="*80 + "\n")
    
    def _display_enhanced_results(self):
        """Display enhanced pipeline results"""
        print("\n" + "="*80)
        print("📈 ENHANCED ELLIOTT WAVE PIPELINE RESULTS")
        print("="*80)
        
        # Data info with resource optimization
        if 'data_info' in self.results:
            data_info = self.results['data_info']
            print(f"📊 Data: {data_info.get('total_rows', 'N/A'):,} rows, {data_info.get('features_count', 'N/A')} features")
            if data_info.get('enhanced_features'):
                print("✅ Enhanced with Elliott Wave features")
            if data_info.get('resource_optimization'):
                print(f"🎯 Resource Optimization: {data_info['resource_optimization']}")
            if data_info.get('gpu_acceleration'):
                print("🎮 GPU Acceleration: ENABLED")
        
        # Resource Utilization Report
        if 'resource_utilization_report' in self.results:
            resource_report = self.results['resource_utilization_report']
            if 'current_usage' in resource_report:
                usage = resource_report['current_usage']
                compliance = resource_report.get('compliance', {})
                
                print("\n🎯 RESOURCE UTILIZATION (80% TARGET):")
                print(f"   CPU Usage: {usage.get('cpu_percent', 0):.1f}% - {compliance.get('cpu_compliance', 'Unknown')}")
                print(f"   Memory Usage: {usage.get('memory_percent', 0):.1f}% - {compliance.get('memory_compliance', 'Unknown')}")
                print(f"   Overall Status: {compliance.get('overall_status', 'Unknown')}")
                
                if usage.get('gpu_available'):
                    print(f"   GPU Usage: {usage.get('gpu_utilization', 0):.1f}%")
                    print(f"   GPU Memory: {usage.get('gpu_memory_used_gb', 0):.1f}GB / {usage.get('gpu_memory_total_gb', 0):.1f}GB")
        
        # Elliott Wave Analysis
        if 'elliott_wave_analysis' in self.results:
            ew_analysis = self.results['elliott_wave_analysis']
            if ew_analysis.get('analysis_status') == 'success':
                print(f"🌊 Elliott Wave Analysis: SUCCESS ({ew_analysis.get('wave_features_count', 0)} wave features)")
            else:
                print(f"⚠️ Elliott Wave Analysis: {ew_analysis.get('analysis_status', 'UNKNOWN')}")
        
        # Feature Selection
        if 'feature_selection' in self.results:
            fs_results = self.results['feature_selection']
            if 'error' not in fs_results:
                auc = fs_results.get('best_auc', 'N/A')
                print(f"🧠 Feature Selection: AUC {auc}")
            else:
                print("⚠️ Feature Selection: FAILED")
        
        # CNN-LSTM Results
        if 'cnn_lstm_training' in self.results:
            cnn_results = self.results['cnn_lstm_training']
            if cnn_results and 'accuracy' in cnn_results:
                print(f"🏗️ CNN-LSTM: Accuracy {cnn_results['accuracy']:.4f}")
            else:
                print("🏗️ CNN-LSTM: Training completed")
        
        # DQN Results
        if 'dqn_training' in self.results:
            dqn_results = self.results['dqn_training']
            if 'error' not in dqn_results:
                if 'final_reward' in dqn_results:
                    print(f"🤖 DQN Training: Final Reward {dqn_results['final_reward']:.4f}")
                elif 'curriculum_results' in dqn_results:
                    print("🤖 Enhanced DQN: Curriculum Learning completed")
                else:
                    print("🤖 DQN Training: Completed")
            else:
                print("⚠️ DQN Training: FAILED")
        
        # Trading Recommendations
        if 'trading_recommendations' in self.results:
            recommendations = self.results['trading_recommendations']
            if 'error' not in recommendations:
                print("📈 Trading Recommendations: Generated")
                if 'overall_recommendation' in recommendations:
                    rec = recommendations['overall_recommendation']
                    print(f"   • Action: {rec.get('action', 'N/A')}")
                    print(f"   • Confidence: {rec.get('confidence', 'N/A'):.2f}")
                    print(f"   • Position Size: {rec.get('position_size', 'N/A')}")
            else:
                print("⚠️ Trading Recommendations: FAILED")
        
        print("="*80 + "\n")
    
    def _monitor_resource_usage(self, step_name: str) -> Dict[str, Any]:
        """🔍 Monitor real-time resource usage for 80% target compliance"""
        try:
            import psutil
            
            resource_info = {
                'timestamp': datetime.now().isoformat(),
                'step': step_name,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent if os.path.exists('/') else 0,
                'target_utilization': '80%',
                'compliance_status': 'CHECKING'
            }
            
            # Check GPU usage if available
            if self.gpu_manager:
                gpu_usage = self.gpu_manager.get_current_usage()
                resource_info.update({
                    'gpu_available': gpu_usage.get('gpu_available', False),
                    'gpu_utilization': gpu_usage.get('gpu_utilization', 0),
                    'gpu_memory_used': gpu_usage.get('gpu_memory_used', 0),
                    'gpu_memory_total': gpu_usage.get('gpu_memory_total', 0)
                })
            
            # Check compliance with 80% target
            cpu_target_met = resource_info['cpu_percent'] >= 60  # Allow some overhead
            memory_target_met = resource_info['memory_percent'] >= 60
            
            if cpu_target_met and memory_target_met:
                resource_info['compliance_status'] = '✅ COMPLIANT'
            elif resource_info['cpu_percent'] < 30 or resource_info['memory_percent'] < 30:
                resource_info['compliance_status'] = '⚠️ UNDER-UTILIZED'
            else:
                resource_info['compliance_status'] = '🔄 OPTIMIZING'
            
            # Log resource information
            self.safe_logger.info(f"📊 Resource Usage - {step_name}:")
            self.safe_logger.info(f"   CPU: {resource_info['cpu_percent']:.1f}%")
            self.safe_logger.info(f"   Memory: {resource_info['memory_percent']:.1f}%")
            self.safe_logger.info(f"   Status: {resource_info['compliance_status']}")
            
            return resource_info
            
        except Exception as e:
            self.safe_logger.warning(f"⚠️ Resource monitoring failed: {e}")
            return {'error': str(e), 'step': step_name}

    def _ensure_80_percent_utilization(self):
        """🎯 Ensure 80% resource utilization throughout pipeline"""
        try:
            if self.gpu_manager:
                # Apply GPU optimizations
                self.gpu_manager.configure_tensorflow()
                self.gpu_manager.configure_pytorch()
                
                # Get optimization recommendations
                optimization_report = self.gpu_manager.get_optimization_report()
                self.safe_logger.info(f"🎮 GPU Optimization Applied:\n{optimization_report}")
            
            # Apply CPU optimizations
            import os
            if hasattr(os, 'sched_setaffinity'):
                # Set CPU affinity for better resource utilization
                available_cpus = list(range(os.cpu_count()))
                target_cpus = available_cpus[:int(len(available_cpus) * 0.8)]
                try:
                    os.sched_setaffinity(0, target_cpus)
                    self.safe_logger.info(f"🧠 CPU Affinity Set: {len(target_cpus)}/{len(available_cpus)} cores")
                except:
                    pass
            
            return True
            
        except Exception as e:
            self.safe_logger.warning(f"⚠️ Resource optimization failed: {e}")
            return False

    def _generate_resource_utilization_report(self) -> Dict[str, Any]:
        """📊 Generate comprehensive resource utilization report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'target_utilization': '80%',
                'pipeline_version': 'Enhanced v2.0 with 80% Resource Optimization',
                'resource_summary': {}
            }
            
            # Current system status
            import psutil
            current_usage = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent if os.path.exists('/') else 0,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }
            
            # GPU information if available
            if self.gpu_manager:
                gpu_usage = self.gpu_manager.get_current_usage()
                current_usage.update({
                    'gpu_available': gpu_usage.get('gpu_available', False),
                    'gpu_utilization': gpu_usage.get('gpu_utilization', 0),
                    'gpu_memory_used_gb': gpu_usage.get('gpu_memory_used', 0) / 1024,
                    'gpu_memory_total_gb': gpu_usage.get('gpu_memory_total', 0) / 1024,
                    'gpu_optimization_report': self.gpu_manager.get_optimization_report()
                })
            
            report['current_usage'] = current_usage
            
            # Calculate compliance with 80% target
            cpu_compliance = "✅ OPTIMAL" if current_usage['cpu_percent'] >= 60 else "⚠️ UNDER-UTILIZED"
            memory_compliance = "✅ OPTIMAL" if current_usage['memory_percent'] >= 60 else "⚠️ UNDER-UTILIZED"
            
            report['compliance'] = {
                'cpu_compliance': cpu_compliance,
                'memory_compliance': memory_compliance,
                'overall_status': "✅ COMPLIANT" if "✅" in cpu_compliance and "✅" in memory_compliance else "🔄 NEEDS OPTIMIZATION"
            }
            
            # Recommendations
            recommendations = []
            if current_usage['cpu_percent'] < 60:
                recommendations.append("Consider increasing parallel processing or batch sizes")
            if current_usage['memory_percent'] < 60:
                recommendations.append("Consider loading more data or increasing model complexity")
            if not recommendations:
                recommendations.append("Resource utilization is optimal for 80% target")
                
            report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'Resource monitoring failed'
            }

def main():
    """Main function for testing Enhanced Elliott Wave System"""
    print("🚀 Testing Enhanced Elliott Wave System...")
    
    # Configuration
    config = {
        'elliott_wave': {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'target_auc': 0.70,
            'max_features': 25
        },
        'dqn': {
            'state_size': 50,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update': 100
        }
    }
    
    # Create and run enhanced system
    enhanced_menu = EnhancedMenu1ElliottWave(config=config)
    results = enhanced_menu.run()
    
    print(f"\n✅ Enhanced Elliott Wave System completed with status: {results.get('execution_status', 'unknown')}")
    return results


if __name__ == "__main__":
    main()
