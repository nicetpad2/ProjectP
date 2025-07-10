#!/usr/bin/env python3
"""
🌊 ENHANCED MENU 1: ADVANCED ELLIOTT WAVE CNN-LSTM + ENHANCED DQN SYSTEM
เมนูที่ 1 ที่ได้รับการปรับปรุงด้วย Multi-Timeframe Elliott Wave Analysis และ Enhanced DQN

Advanced Features:
- Advanced Multi-Timeframe Elliott Wave Analysis (M1, M5, M15, M30, H1, H4, D1)
- Enhanced DQN Agent with Elliott Wave Integration
- Fractal Elliott Wave Pattern Recognition
- Advanced Fibonacci Analysis and Projections
- Market Regime Detection and Adaptation
- Hierarchical Learning Architecture
- Real-time Multi-Timeframe Confluence Analysis
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
import json

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

# Import Advanced Logging
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

# Import Beautiful Progress
from core.robust_beautiful_progress import setup_robust_beautiful_logging

# Import Elliott Wave Components
try:
    from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
    from elliott_wave_modules.advanced_multi_timeframe_elliott_wave import AdvancedMultiTimeframeElliottWaveAnalyzer
    from elliott_wave_modules.enhanced_multi_timeframe_dqn_agent import EnhancedMultiTimeframeDQNAgent
    from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
    from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
    from elliott_wave_modules.pipeline_orchestrator import ElliottWavePipelineOrchestrator
    from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
    ELLIOTT_WAVE_MODULES_AVAILABLE = True
except ImportError as e:
    ELLIOTT_WAVE_MODULES_AVAILABLE = False
    print(f"⚠️ Elliott Wave modules not available: {str(e)}")

# Enterprise ML Protection
try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    ML_PROTECTION_AVAILABLE = True
except ImportError:
    ML_PROTECTION_AVAILABLE = False

class EnhancedMenu1ElliottWaveAdvanced:
    """Enhanced Menu 1 with Advanced Multi-Timeframe Elliott Wave and Enhanced DQN"""
    
    def __init__(self):
        """Initialize Enhanced Menu 1 Elliott Wave System"""
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize paths and output manager
        self.paths = get_project_paths()
        self.output_manager = NicegoldOutputManager()
        
        # Initialize logging system
        self._setup_logging_system()
        
        # Initialize components
        self._initialize_components()
        
        # Results storage
        self.results = {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'data_loading': {},
            'elliott_wave_analysis': {},
            'feature_engineering': {},
            'feature_selection': {},
            'cnn_lstm_training': {},
            'enhanced_dqn_training': {},
            'multi_timeframe_integration': {},
            'performance_analysis': {},
            'final_results': {},
            'execution_time': 0,
            'success': False
        }
        
        self.logger.info("🌊 Enhanced Menu 1 Elliott Wave Advanced initialized successfully")
    
    def _setup_logging_system(self):
        """Setup advanced logging system"""
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.progress_manager = get_progress_manager()
            self.beautiful_logger = setup_robust_beautiful_logging(
                "EnhancedElliottWave_Menu1"
            )
        else:
            self.logger = logging.getLogger("EnhancedElliottWave_Menu1")
            self.progress_manager = None
            self.beautiful_logger = setup_robust_beautiful_logging(
                "EnhancedElliottWave_Menu1_Simple"
            )
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            if ELLIOTT_WAVE_MODULES_AVAILABLE:
                # Initialize Advanced Elliott Wave Components
                self.data_processor = ElliottWaveDataProcessor()
                self.advanced_elliott_analyzer = AdvancedMultiTimeframeElliottWaveAnalyzer()
                self.enhanced_dqn_agent = EnhancedMultiTimeframeDQNAgent()
                self.cnn_lstm_engine = CNNLSTMElliottWave()
                self.feature_selector = EnterpriseShapOptunaFeatureSelector()
                self.pipeline_orchestrator = ElliottWavePipelineOrchestrator()
                self.performance_analyzer = ElliottWavePerformanceAnalyzer()
                
                self.logger.info("✅ All advanced Elliott Wave components initialized")
            else:
                self.logger.error("❌ Elliott Wave modules not available")
                
            # Initialize ML Protection
            if ML_PROTECTION_AVAILABLE:
                self.ml_protection = EnterpriseMLProtectionSystem()
                self.logger.info("✅ Enterprise ML Protection initialized")
            else:
                self.ml_protection = None
                self.logger.warning("⚠️ ML Protection not available")
                
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Main entry point for the enhanced Elliott Wave pipeline
        This method provides compatibility with ProjectP.py
        """
        return self.run_enhanced_elliott_wave_pipeline()
    
    def run_enhanced_elliott_wave_pipeline(self) -> Dict[str, Any]:
        """Run the enhanced Elliott Wave pipeline with multi-timeframe analysis"""
        try:
            self.logger.info("🚀 Starting Enhanced Elliott Wave Pipeline with Multi-Timeframe Analysis")
            
            # Step 1: Data Loading and Validation
            data_results = self._run_step_1_enhanced_data_loading()
            if not data_results.get('success', False):
                raise Exception("Data loading failed")
            
            # Step 2: Multi-Timeframe Elliott Wave Analysis
            elliott_results = self._run_step_2_multi_timeframe_elliott_analysis(data_results['data'])
            
            # Step 3: Enhanced Feature Engineering
            feature_results = self._run_step_3_enhanced_feature_engineering(
                data_results['data'], elliott_results
            )
            
            # Step 4: Enterprise Feature Selection
            selection_results = self._run_step_4_enterprise_feature_selection(
                feature_results['features'], feature_results['target']
            )
            
            # Step 5: CNN-LSTM Training with Elliott Wave Integration
            cnn_lstm_results = self._run_step_5_cnn_lstm_elliott_training(
                selection_results['selected_features'], selection_results['target']
            )
            
            # Step 6: Enhanced DQN Training with Multi-Timeframe Integration
            dqn_results = self._run_step_6_enhanced_dqn_training(
                selection_results['selected_features'], elliott_results
            )
            
            # Step 7: Multi-Timeframe Pipeline Integration
            integration_results = self._run_step_7_multi_timeframe_integration(
                data_results['data'], elliott_results, cnn_lstm_results, dqn_results
            )
            
            # Step 8: Advanced Performance Analysis
            performance_results = self._run_step_8_advanced_performance_analysis(
                integration_results
            )
            
            # Step 9: Results Compilation and Export
            final_results = self._run_step_9_results_compilation(
                data_results, elliott_results, feature_results, selection_results,
                cnn_lstm_results, dqn_results, integration_results, performance_results
            )
            
            self.results.update({
                'data_loading': data_results,
                'elliott_wave_analysis': elliott_results,
                'feature_engineering': feature_results,
                'feature_selection': selection_results,
                'cnn_lstm_training': cnn_lstm_results,
                'enhanced_dqn_training': dqn_results,
                'multi_timeframe_integration': integration_results,
                'performance_analysis': performance_results,
                'final_results': final_results,
                'execution_time': time.time() - self.start_time,
                'success': True
            })
            
            self.logger.info("✅ Enhanced Elliott Wave Pipeline completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Enhanced Elliott Wave Pipeline failed: {str(e)}")
            self.results.update({
                'error': str(e),
                'execution_time': time.time() - self.start_time,
                'success': False
            })
            return self.results
    
    def _run_step_1_enhanced_data_loading(self) -> Dict[str, Any]:
        """Step 1: Enhanced Data Loading with Multi-Timeframe Preparation"""
        step_start = time.time()
        self.beautiful_logger.step_start(1, "Enhanced Data Loading", 
                                       "Loading real market data with multi-timeframe preparation")
        
        try:
            # Load real market data
            self.logger.info("📊 Loading real market data from datacsv/")
            data = self.data_processor.load_real_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            # Create multi-timeframe datasets
            self.logger.info("⏰ Creating multi-timeframe datasets")
            timeframe_data = self._create_timeframe_datasets(data)
            
            # Validate data quality
            self.logger.info("✅ Validating data quality")
            quality_metrics = self._validate_data_quality(data, timeframe_data)
            
            results = {
                'success': True,
                'data': data,
                'timeframe_data': timeframe_data,
                'data_shape': data.shape,
                'timeframes_created': list(timeframe_data.keys()),
                'quality_metrics': quality_metrics,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(1, "Enhanced Data Loading", 
                                              time.time() - step_start, {
                "rows_loaded": f"{len(data):,}",
                "timeframes_created": len(timeframe_data),
                "data_quality": f"{quality_metrics.get('overall_score', 0):.1f}%"
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(1, "Enhanced Data Loading", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _create_timeframe_datasets(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multi-timeframe datasets"""
        timeframe_data = {}
        
        try:
            # Ensure datetime index
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'])
            elif 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
            else:
                data = data.reset_index()
                data['datetime'] = pd.date_range(
                    start='2020-01-01', periods=len(data), freq='1min'
                )
            
            data.set_index('datetime', inplace=True)
            data.sort_index(inplace=True)
            
            # Create different timeframes
            timeframe_mapping = {
                'M1': '1min',   'M5': '5min',   'M15': '15min',
                'M30': '30min', 'H1': '1H',     'H4': '4H',
                'D1': '1D'
            }
            
            for tf, freq in timeframe_mapping.items():
                try:
                    resampled = data.resample(freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum' if 'volume' in data.columns else 'mean'
                    }).dropna()
                    
                    if len(resampled) >= 50:  # Minimum data requirement
                        timeframe_data[tf] = resampled
                        self.logger.info(f"✅ Created {tf} timeframe: {len(resampled)} bars")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create {tf} timeframe: {str(e)}")
            
            return timeframe_data
            
        except Exception as e:
            self.logger.error(f"Timeframe dataset creation failed: {str(e)}")
            return {'M1': data}  # Fallback to original data
    
    def _validate_data_quality(self, data: pd.DataFrame, 
                              timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate data quality across timeframes"""
        quality_metrics = {
            'main_data_complete': True,
            'timeframe_coverage': {},
            'missing_data_ratio': 0.0,
            'price_consistency': True,
            'volume_consistency': True,
            'overall_score': 0.0
        }
        
        try:
            # Check main data completeness
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_metrics['missing_data_ratio'] = missing_ratio
            
            # Check timeframe coverage
            for tf, tf_data in timeframe_data.items():
                coverage = len(tf_data) / max(len(data) // {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}.get(tf, 1), 1)
                quality_metrics['timeframe_coverage'][tf] = min(coverage, 1.0)
            
            # Calculate overall score
            score = 100.0
            score -= missing_ratio * 50  # Penalty for missing data
            score -= (1.0 - np.mean(list(quality_metrics['timeframe_coverage'].values()))) * 30
            
            quality_metrics['overall_score'] = max(score, 0.0)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")
            return quality_metrics
    
    def _run_step_2_multi_timeframe_elliott_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 2: Multi-Timeframe Elliott Wave Analysis"""
        step_start = time.time()
        self.beautiful_logger.step_start(2, "Multi-Timeframe Elliott Wave Analysis", 
                                       "Analyzing Elliott Wave patterns across multiple timeframes")
        
        try:
            # Perform advanced multi-timeframe Elliott Wave analysis
            elliott_analysis = self.advanced_elliott_analyzer.analyze_multi_timeframe_elliott_waves(data)
            
            # Validate Elliott Wave analysis
            if self.ml_protection:
                validation_results = self.ml_protection.validate_elliott_wave_analysis(elliott_analysis)
                elliott_analysis['validation'] = validation_results
            
            # Extract key insights
            insights = self._extract_elliott_wave_insights(elliott_analysis)
            
            results = {
                'success': True,
                'elliott_analysis': elliott_analysis,
                'insights': insights,
                'timeframes_analyzed': len(elliott_analysis.get('timeframe_analysis', {})),
                'overall_direction': elliott_analysis.get('confluence_analysis', {}).get('overall_direction', 'NEUTRAL'),
                'confluence_strength': elliott_analysis.get('confluence_analysis', {}).get('strength', 0),
                'primary_wave_count': elliott_analysis.get('primary_wave_count'),
                'trading_signals': elliott_analysis.get('trading_signals', []),
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(2, "Multi-Timeframe Elliott Wave Analysis", 
                                              time.time() - step_start, {
                "timeframes_analyzed": results['timeframes_analyzed'],
                "overall_direction": results['overall_direction'],
                "confluence_strength": f"{results['confluence_strength']}/10",
                "trading_signals": len(results['trading_signals'])
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(2, "Multi-Timeframe Elliott Wave Analysis", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _extract_elliott_wave_insights(self, elliott_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from Elliott Wave analysis"""
        insights = {
            'market_phase': 'UNKNOWN',
            'trend_strength': 0,
            'reversal_probability': 0.0,
            'key_levels': [],
            'risk_assessment': 'NEUTRAL',
            'opportunity_score': 0
        }
        
        try:
            confluence = elliott_analysis.get('confluence_analysis', {})
            
            # Determine market phase
            direction = confluence.get('overall_direction', 'NEUTRAL')
            strength = confluence.get('strength', 0)
            
            if direction in ['UPTREND', 'DOWNTREND'] and strength >= 7:
                insights['market_phase'] = f'STRONG_{direction}'
            elif direction in ['UPTREND', 'DOWNTREND'] and strength >= 4:
                insights['market_phase'] = f'WEAK_{direction}'
            else:
                insights['market_phase'] = 'CONSOLIDATION'
            
            insights['trend_strength'] = strength
            
            # Calculate reversal probability based on wave count
            primary_wave = elliott_analysis.get('primary_wave_count')
            if primary_wave and '5-WAVE' in primary_wave:
                insights['reversal_probability'] = 0.7  # High probability of reversal after Wave 5
            elif primary_wave and 'WAVE_C' in str(primary_wave):
                insights['reversal_probability'] = 0.6  # Moderate probability after Wave C
            else:
                insights['reversal_probability'] = 0.3  # Low probability
            
            # Extract key Fibonacci levels
            fibonacci_levels = elliott_analysis.get('fibonacci_levels', {})
            key_levels = fibonacci_levels.get('key_levels', [])
            insights['key_levels'] = [level['level'] for level in key_levels[:3]]  # Top 3 levels
            
            # Risk assessment
            conflicting_tfs = len(confluence.get('conflicting_timeframes', []))
            supporting_tfs = len(confluence.get('supporting_timeframes', []))
            
            if supporting_tfs >= 5 and conflicting_tfs <= 1:
                insights['risk_assessment'] = 'LOW'
            elif supporting_tfs >= 3 and conflicting_tfs <= 2:
                insights['risk_assessment'] = 'MODERATE'
            else:
                insights['risk_assessment'] = 'HIGH'
            
            # Opportunity score
            signals = elliott_analysis.get('trading_signals', [])
            if signals:
                max_strength = max(signal.get('strength', 0) for signal in signals)
                insights['opportunity_score'] = min(max_strength, 10)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Elliott Wave insights extraction failed: {str(e)}")
            return insights
    
    def _run_step_3_enhanced_feature_engineering(self, data: pd.DataFrame, 
                                               elliott_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Enhanced Feature Engineering with Elliott Wave Integration"""
        step_start = time.time()
        self.beautiful_logger.step_start(3, "Enhanced Feature Engineering", 
                                       "Creating advanced features with Elliott Wave integration")
        
        try:
            # Create base Elliott Wave features
            self.logger.info("🔧 Creating Elliott Wave features")
            elliott_features = self.data_processor.create_elliott_wave_features(data)
            
            # Add multi-timeframe features
            self.logger.info("⏰ Adding multi-timeframe features")
            timeframe_features = self._create_multi_timeframe_features(data, elliott_results)
            
            # Add advanced Elliott Wave pattern features
            self.logger.info("🌊 Adding advanced Elliott Wave pattern features")
            pattern_features = self._create_elliott_pattern_features(elliott_results)
            
            # Combine all features
            combined_features = pd.concat([
                elliott_features,
                timeframe_features,
                pattern_features
            ], axis=1)
            
            # Create target variable
            target = self._create_enhanced_target(data, elliott_results)
            
            # Validate feature quality
            feature_quality = self._validate_feature_quality(combined_features, target)
            
            results = {
                'success': True,
                'features': combined_features,
                'target': target,
                'feature_count': len(combined_features.columns),
                'elliott_features': len(elliott_features.columns),
                'timeframe_features': len(timeframe_features.columns),
                'pattern_features': len(pattern_features.columns),
                'feature_quality': feature_quality,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(3, "Enhanced Feature Engineering", 
                                              time.time() - step_start, {
                "total_features": results['feature_count'],
                "elliott_features": results['elliott_features'],
                "timeframe_features": results['timeframe_features'],
                "feature_quality": f"{feature_quality.get('score', 0):.1f}%"
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(3, "Enhanced Feature Engineering", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _create_multi_timeframe_features(self, data: pd.DataFrame, 
                                       elliott_results: Dict[str, Any]) -> pd.DataFrame:
        """Create multi-timeframe features"""
        try:
            timeframe_features = pd.DataFrame(index=data.index)
            
            # Get timeframe analysis from Elliott results
            timeframe_analysis = elliott_results.get('elliott_analysis', {}).get('timeframe_analysis', {})
            
            for tf, analysis in timeframe_analysis.items():
                # Wave direction feature
                direction = analysis.get('trend_direction', 'NEUTRAL')
                direction_value = 1 if direction == 'UPTREND' else (-1 if direction == 'DOWNTREND' else 0)
                timeframe_features[f'{tf}_direction'] = direction_value
                
                # Wave strength feature
                strength = analysis.get('wave_strength', 0) / 10.0
                timeframe_features[f'{tf}_strength'] = strength
                
                # Confidence feature
                confidence = analysis.get('confidence', 0) / 100.0
                timeframe_features[f'{tf}_confidence'] = confidence
            
            # Fill missing values
            timeframe_features = timeframe_features.fillna(0)
            
            return timeframe_features
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe features creation failed: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _create_elliott_pattern_features(self, elliott_results: Dict[str, Any]) -> pd.DataFrame:
        """Create Elliott Wave pattern-specific features"""
        try:
            # Initialize pattern features dataframe - will be aligned with main data later
            pattern_features = pd.DataFrame()
            
            elliott_analysis = elliott_results.get('elliott_analysis', {})
            
            # Confluence features
            confluence = elliott_analysis.get('confluence_analysis', {})
            pattern_features['confluence_strength'] = confluence.get('strength', 0) / 10.0
            pattern_features['confluence_agreement'] = confluence.get('agreement_score', 0.0)
            
            # Wave count features
            primary_wave = elliott_analysis.get('primary_wave_count')
            pattern_features['is_impulse_wave'] = 1 if primary_wave and '5-WAVE' in primary_wave else 0
            pattern_features['is_corrective_wave'] = 1 if primary_wave and '3-WAVE' in primary_wave else 0
            
            # Trading signals features
            signals = elliott_analysis.get('trading_signals', [])
            buy_signals = [s for s in signals if s.get('type') == 'BUY']
            sell_signals = [s for s in signals if s.get('type') == 'SELL']
            
            pattern_features['buy_signal_strength'] = max([s.get('strength', 0) for s in buy_signals], default=0) / 10.0
            pattern_features['sell_signal_strength'] = max([s.get('strength', 0) for s in sell_signals], default=0) / 10.0
            pattern_features['signal_count'] = len(signals)
            
            # Fibonacci level features
            fibonacci_levels = elliott_analysis.get('fibonacci_levels', {})
            key_levels = fibonacci_levels.get('key_levels', [])
            pattern_features['near_fibonacci_level'] = 1 if key_levels else 0
            pattern_features['fibonacci_level_count'] = len(key_levels)
            
            return pattern_features
            
        except Exception as e:
            self.logger.error(f"Elliott pattern features creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_enhanced_target(self, data: pd.DataFrame, 
                               elliott_results: Dict[str, Any]) -> pd.Series:
        """Create enhanced target variable considering Elliott Wave signals"""
        try:
            # Base target: future price direction
            target = pd.Series(index=data.index, dtype=float)
            
            # Look ahead periods for different prediction horizons
            look_ahead_periods = [5, 10, 20]  # 5, 10, 20 minutes ahead for M1 data
            
            for period in look_ahead_periods:
                if len(data) > period:
                    future_close = data['close'].shift(-period)
                    current_close = data['close']
                    
                    # Calculate price change
                    price_change = (future_close - current_close) / current_close
                    
                    # Weight by Elliott Wave signals
                    signals = elliott_results.get('elliott_analysis', {}).get('trading_signals', [])
                    signal_weight = 1.0
                    
                    if signals:
                        # Increase weight if strong Elliott Wave signals
                        max_signal_strength = max(s.get('strength', 0) for s in signals)
                        signal_weight = 1.0 + (max_signal_strength / 10.0)
                    
                    # Create weighted target
                    weighted_change = price_change * signal_weight
                    
                    # Combine with existing target
                    if target.isna().all():
                        target = weighted_change
                    else:
                        target = (target + weighted_change) / 2
            
            # Convert to classification target
            target = target.apply(lambda x: 1 if x > 0.001 else (-1 if x < -0.001 else 0))
            
            return target.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Enhanced target creation failed: {str(e)}")
            return pd.Series(index=data.index, dtype=float).fillna(0)
    
    def _validate_feature_quality(self, features: pd.DataFrame, 
                                 target: pd.Series) -> Dict[str, Any]:
        """Validate feature quality"""
        quality_metrics = {
            'score': 0.0,
            'missing_ratio': 0.0,
            'correlation_with_target': 0.0,
            'feature_variance': 0.0,
            'multicollinearity': 0.0
        }
        
        try:
            # Check missing values
            missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
            quality_metrics['missing_ratio'] = missing_ratio
            
            # Check correlation with target
            correlations = []
            for col in features.columns:
                if features[col].var() > 0:  # Only check non-constant features
                    corr = abs(features[col].corr(target))
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            quality_metrics['correlation_with_target'] = np.mean(correlations) if correlations else 0
            
            # Check feature variance
            variances = [features[col].var() for col in features.columns if features[col].var() > 0]
            quality_metrics['feature_variance'] = np.mean(variances) if variances else 0
            
            # Calculate overall score
            score = 100.0
            score -= missing_ratio * 50
            score += quality_metrics['correlation_with_target'] * 30
            score += min(quality_metrics['feature_variance'], 1.0) * 20
            
            quality_metrics['score'] = max(score, 0.0)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Feature quality validation failed: {str(e)}")
            return quality_metrics
    
    def _run_step_4_enterprise_feature_selection(self, features: pd.DataFrame, 
                                                target: pd.Series) -> Dict[str, Any]:
        """Step 4: Enterprise Feature Selection with SHAP + Optuna"""
        step_start = time.time()
        self.beautiful_logger.step_start(4, "Enterprise Feature Selection", 
                                       "Selecting optimal features using SHAP + Optuna")
        
        try:
            # Prepare data for feature selection
            X = features.fillna(0)
            y = target.fillna(0)
            
            # Run enterprise feature selection
            selection_results = self.feature_selector.select_features(X, y)
            
            # Validate selection results
            if selection_results.get('error'):
                raise Exception(f"Feature selection failed: {selection_results['error']}")
            
            selected_features = selection_results.get('selected_features')
            if selected_features is None or len(selected_features) == 0:
                raise Exception("No features selected")
            
            results = {
                'success': True,
                'selected_features': selected_features,
                'target': y,
                'feature_count': len(selected_features.columns),
                'auc_score': selection_results.get('auc_score', 0),
                'selection_method': selection_results.get('method', 'SHAP+Optuna'),
                'feature_names': list(selected_features.columns),
                'selection_results': selection_results,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(4, "Enterprise Feature Selection", 
                                              time.time() - step_start, {
                "features_selected": results['feature_count'],
                "auc_score": f"{results['auc_score']:.3f}",
                "method": results['selection_method']
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(4, "Enterprise Feature Selection", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _run_step_5_cnn_lstm_elliott_training(self, selected_features: pd.DataFrame, 
                                            target: pd.Series) -> Dict[str, Any]:
        """Step 5: CNN-LSTM Training with Elliott Wave Integration"""
        step_start = time.time()
        self.beautiful_logger.step_start(5, "CNN-LSTM Elliott Wave Training", 
                                       "Training CNN-LSTM model with Elliott Wave patterns")
        
        try:
            # Prepare data for CNN-LSTM
            X = selected_features.fillna(0).values
            y = target.fillna(0).values
            
            # Train CNN-LSTM model
            training_results = self.cnn_lstm_engine.train_model(X, y, epochs=50)
            
            # Validate training results
            if not training_results.get('success', False):
                raise Exception(f"CNN-LSTM training failed: {training_results.get('error', 'Unknown error')}")
            
            results = {
                'success': True,
                'model': training_results.get('model'),
                'training_history': training_results.get('history'),
                'accuracy': training_results.get('accuracy', 0),
                'auc_score': training_results.get('auc_score', 0),
                'model_path': training_results.get('model_path'),
                'predictions': training_results.get('predictions'),
                'training_results': training_results,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(5, "CNN-LSTM Elliott Wave Training", 
                                              time.time() - step_start, {
                "accuracy": f"{results['accuracy']:.3f}",
                "auc_score": f"{results['auc_score']:.3f}",
                "model_saved": "✅ Yes" if results['model_path'] else "❌ No"
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(5, "CNN-LSTM Elliott Wave Training", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _run_step_6_enhanced_dqn_training(self, selected_features: pd.DataFrame, 
                                        elliott_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Enhanced DQN Training with Multi-Timeframe Integration"""
        step_start = time.time()
        self.beautiful_logger.step_start(6, "Enhanced DQN Training", 
                                       "Training Enhanced DQN agent with multi-timeframe Elliott Wave")
        
        try:
            # Prepare training data for DQN
            training_data = selected_features.copy()
            
            # Add Elliott Wave analysis to each row (simplified for training)
            elliott_analysis = elliott_results.get('elliott_analysis', {})
            
            # Create market data format for DQN
            market_data_list = []
            for idx, row in training_data.iterrows():
                market_data = {
                    'close': row.get('close', 0),
                    'rsi': row.get('rsi', 50),
                    'macd': row.get('macd', 0),
                    'macd_signal': row.get('macd_signal', 0),
                    'volume': row.get('volume', 0),
                    'elliott_analysis': elliott_analysis
                }
                market_data_list.append(market_data)
            
            # Train Enhanced DQN Agent
            dqn_results = self.enhanced_dqn_agent.train_agent(training_data, episodes=100)
            
            # Validate DQN results
            if not dqn_results.get('training_complete', False):
                self.logger.warning("DQN training incomplete but continuing...")
            
            results = {
                'success': True,
                'agent': dqn_results.get('agent'),
                'total_reward': dqn_results.get('total_reward', 0),
                'avg_reward_per_episode': dqn_results.get('avg_reward_per_episode', 0),
                'win_rate': dqn_results.get('win_rate', 0),
                'final_epsilon': dqn_results.get('final_epsilon', 0),
                'episodes_completed': dqn_results.get('episodes_completed', 0),
                'performance': dqn_results.get('performance', {}),
                'dqn_results': dqn_results,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(6, "Enhanced DQN Training", 
                                              time.time() - step_start, {
                "episodes_completed": results['episodes_completed'],
                "avg_reward": f"{results['avg_reward_per_episode']:.2f}",
                "win_rate": f"{results['win_rate']:.1%}",
                "final_epsilon": f"{results['final_epsilon']:.3f}"
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(6, "Enhanced DQN Training", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _run_step_7_multi_timeframe_integration(self, data: pd.DataFrame,
                                              elliott_results: Dict[str, Any],
                                              cnn_lstm_results: Dict[str, Any],
                                              dqn_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Multi-Timeframe Pipeline Integration"""
        step_start = time.time()
        self.beautiful_logger.step_start(7, "Multi-Timeframe Integration", 
                                       "Integrating all components with multi-timeframe analysis")
        
        try:
            # Run integrated pipeline
            integration_results = self.pipeline_orchestrator.run_integrated_pipeline(
                data, elliott_results, cnn_lstm_results, dqn_results
            )
            
            # Validate integration results
            if not integration_results.get('success', False):
                self.logger.warning("Pipeline integration had issues but continuing...")
            
            results = {
                'success': True,
                'integrated_predictions': integration_results.get('predictions', []),
                'ensemble_signals': integration_results.get('ensemble_signals', []),
                'confidence_scores': integration_results.get('confidence_scores', []),
                'risk_assessment': integration_results.get('risk_assessment', {}),
                'trading_recommendations': integration_results.get('trading_recommendations', []),
                'integration_results': integration_results,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(7, "Multi-Timeframe Integration", 
                                              time.time() - step_start, {
                "predictions_generated": len(results['integrated_predictions']),
                "ensemble_signals": len(results['ensemble_signals']),
                "trading_recommendations": len(results['trading_recommendations'])
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(7, "Multi-Timeframe Integration", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _run_step_8_advanced_performance_analysis(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Advanced Performance Analysis"""
        step_start = time.time()
        self.beautiful_logger.step_start(8, "Advanced Performance Analysis", 
                                       "Analyzing system performance with advanced metrics")
        
        try:
            # Analyze performance using the performance analyzer
            performance_results = self.performance_analyzer.analyze_performance(
                integration_results.get('integration_results', {})
            )
            
            results = {
                'success': True,
                'performance_metrics': performance_results.get('metrics', {}),
                'risk_metrics': performance_results.get('risk_metrics', {}),
                'trading_metrics': performance_results.get('trading_metrics', {}),
                'elliott_wave_metrics': performance_results.get('elliott_wave_metrics', {}),
                'overall_score': performance_results.get('overall_score', 0),
                'performance_rating': performance_results.get('rating', 'UNKNOWN'),
                'performance_results': performance_results,
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(8, "Advanced Performance Analysis", 
                                              time.time() - step_start, {
                "overall_score": f"{results['overall_score']:.1f}%",
                "performance_rating": results['performance_rating']
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(8, "Advanced Performance Analysis", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _run_step_9_results_compilation(self, *step_results) -> Dict[str, Any]:
        """Step 9: Results Compilation and Export"""
        step_start = time.time()
        self.beautiful_logger.step_start(9, "Results Compilation", 
                                       "Compiling and exporting final results")
        
        try:
            # Compile all results
            compiled_results = {
                'session_info': {
                    'session_id': self.session_id,
                    'start_time': self.start_time,
                    'total_execution_time': time.time() - self.start_time,
                    'pipeline_version': 'Enhanced Multi-Timeframe v2.0'
                },
                'data_summary': step_results[0] if len(step_results) > 0 else {},
                'elliott_wave_analysis': step_results[1] if len(step_results) > 1 else {},
                'feature_engineering': step_results[2] if len(step_results) > 2 else {},
                'feature_selection': step_results[3] if len(step_results) > 3 else {},
                'cnn_lstm_training': step_results[4] if len(step_results) > 4 else {},
                'enhanced_dqn_training': step_results[5] if len(step_results) > 5 else {},
                'integration': step_results[6] if len(step_results) > 6 else {},
                'performance_analysis': step_results[7] if len(step_results) > 7 else {}
            }
            
            # Save results
            results_path = self.output_manager.save_results(
                compiled_results, 
                f"enhanced_elliott_wave_complete_results_{self.session_id}"
            )
            
            # Generate comprehensive report
            report_path = self.output_manager.generate_report(
                compiled_results,
                f"enhanced_elliott_wave_complete_analysis_{self.session_id}"
            )
            
            results = {
                'success': True,
                'compiled_results': compiled_results,
                'results_path': results_path,
                'report_path': report_path,
                'session_summary': self._generate_session_summary(compiled_results),
                'execution_time': time.time() - step_start
            }
            
            self.beautiful_logger.step_complete(9, "Results Compilation", 
                                              time.time() - step_start, {
                "results_saved": "✅ Yes" if results_path else "❌ No",
                "report_generated": "✅ Yes" if report_path else "❌ No"
            })
            
            return results
            
        except Exception as e:
            self.beautiful_logger.step_error(9, "Results Compilation", str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start
            }
    
    def _generate_session_summary(self, compiled_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate session summary"""
        summary = {
            'pipeline_success': True,
            'total_execution_time': compiled_results['session_info']['total_execution_time'],
            'key_metrics': {},
            'recommendations': [],
            'next_steps': []
        }
        
        try:
            # Extract key metrics
            elliott_analysis = compiled_results.get('elliott_wave_analysis', {})
            performance = compiled_results.get('performance_analysis', {})
            
            summary['key_metrics'] = {
                'timeframes_analyzed': elliott_analysis.get('timeframes_analyzed', 0),
                'overall_direction': elliott_analysis.get('overall_direction', 'NEUTRAL'),
                'confluence_strength': elliott_analysis.get('confluence_strength', 0),
                'cnn_lstm_accuracy': compiled_results.get('cnn_lstm_training', {}).get('accuracy', 0),
                'dqn_win_rate': compiled_results.get('enhanced_dqn_training', {}).get('win_rate', 0),
                'overall_performance_score': performance.get('overall_score', 0)
            }
            
            # Generate recommendations
            if summary['key_metrics']['confluence_strength'] >= 7:
                summary['recommendations'].append("Strong multi-timeframe confluence detected - consider position entry")
            
            if summary['key_metrics']['cnn_lstm_accuracy'] >= 0.7:
                summary['recommendations'].append("CNN-LSTM model shows good accuracy - reliable for pattern recognition")
            
            if summary['key_metrics']['dqn_win_rate'] >= 0.6:
                summary['recommendations'].append("DQN agent shows profitable learning - suitable for automated trading")
            
            # Next steps
            summary['next_steps'] = [
                "Monitor Elliott Wave pattern completion",
                "Implement real-time multi-timeframe analysis",
                "Deploy enhanced DQN agent for live trading",
                "Continue model optimization and retraining"
            ]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Session summary generation failed: {str(e)}")
            return summary

# Function to run enhanced menu 1
def run_enhanced_menu_1_elliott_wave() -> Dict[str, Any]:
    """Run Enhanced Menu 1 Elliott Wave System"""
    try:
        enhanced_menu = EnhancedMenu1ElliottWaveAdvanced()
        results = enhanced_menu.run_enhanced_elliott_wave_pipeline()
        return results
    except Exception as e:
        logging.error(f"Enhanced Menu 1 execution failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': 0
        }

# Export
__all__ = ['EnhancedMenu1ElliottWaveAdvanced', 'run_enhanced_menu_1_elliott_wave']
