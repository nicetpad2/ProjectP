#!/usr/bin/env python3
"""
🎼 ELLIOTT WAVE PIPELINE ORCHESTRATOR
ตัวควบคุมการทำงานของ Pipeline Elliott Wave แบบครบวงจร

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
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback
import os
import json

class ElliottWavePipelineOrchestrator:
    """ตัวควบคุม Pipeline Elliott Wave ระดับ Enterprise"""
    
    def __init__(self, data_processor, cnn_lstm_engine, dqn_agent, feature_selector, ml_protection=None, config: Dict = None, logger: logging.Logger = None):
        self.data_processor = data_processor
        self.cnn_lstm_engine = cnn_lstm_engine
        self.dqn_agent = dqn_agent
        self.feature_selector = feature_selector
        self.ml_protection = ml_protection  # Enterprise ML Protection System
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Pipeline state
        self.pipeline_state = {
            'stage': 'initialized',
            'progress': 0,
            'start_time': None,
            'end_time': None,
            'errors': [],
            'warnings': []
        }
        
        # Results storage
        self.pipeline_results = {}
    
    def execute_full_pipeline(self) -> Dict[str, Any]:
        """ดำเนินการ Pipeline แบบครบวงจร"""
        try:
            self.logger.info("🚀 Starting Elliott Wave Full Pipeline Execution...")
            self.pipeline_state['start_time'] = datetime.now()
            
            # Pipeline Stages with Enterprise Protection
            stages = [
                ('data_loading', self._stage_1_data_loading),
                ('data_preprocessing', self._stage_2_data_preprocessing),
                ('enterprise_protection_analysis', self._stage_2b_enterprise_protection_analysis),  # New protection stage
                ('feature_engineering', self._stage_3_feature_engineering),
                ('feature_selection', self._stage_4_feature_selection),
                ('pre_training_validation', self._stage_4b_pre_training_validation),  # New validation stage
                ('cnn_lstm_training', self._stage_5_cnn_lstm_training),
                ('dqn_training', self._stage_6_dqn_training),
                ('post_training_protection', self._stage_6b_post_training_protection),  # New protection stage
                ('system_integration', self._stage_7_system_integration),
                ('quality_validation', self._stage_8_quality_validation),
                ('final_protection_report', self._stage_8b_final_protection_report),  # New final report
                ('results_compilation', self._stage_9_results_compilation)
            ]
            
            # Execute stages
            for i, (stage_name, stage_function) in enumerate(stages):
                try:
                    self.logger.info(f"📊 Stage {i+1}/9: {stage_name.replace('_', ' ').title()}")
                    self.pipeline_state['stage'] = stage_name
                    self.pipeline_state['progress'] = int((i / len(stages)) * 100)
                    
                    stage_result = stage_function()
                    self.pipeline_results[stage_name] = stage_result
                    
                    # Check if stage failed
                    if not stage_result.get('success', False):
                        raise Exception(f"Stage {stage_name} failed: {stage_result.get('error', 'Unknown error')}")
                    
                    self.logger.info(f"✅ Stage {i+1}/9 completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"❌ Stage {stage_name} failed: {str(e)}")
                    self.pipeline_state['errors'].append(f"{stage_name}: {str(e)}")
                    
                    # Try to continue with next stage if possible
                    if self._is_critical_stage(stage_name):
                        raise
                    else:
                        self.pipeline_state['warnings'].append(f"Non-critical stage {stage_name} failed, continuing...")
                        continue
            
            # Pipeline completion
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['progress'] = 100
            self.pipeline_state['stage'] = 'completed'
            
            # Compile final results
            final_results = self._compile_final_results()
            
            self.logger.info("✅ Elliott Wave Full Pipeline Execution Completed Successfully!")
            return final_results
            
        except Exception as e:
            self.logger.error(f"💥 Pipeline execution failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Return error results
            return {
                'success': False,
                'error': str(e),
                'pipeline_state': self.pipeline_state,
                'partial_results': self.pipeline_results
            }
    
    def _stage_1_data_loading(self) -> Dict[str, Any]:
        """Stage 1: โหลดและตรวจสอบข้อมูล"""
        try:
            self.logger.info("📂 Loading real market data...")
            
            # Load data
            data = self.data_processor.load_real_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No data loaded or data is empty")
            
            # Validate data quality
            data_quality = self.data_processor.get_data_quality_report(data)
            
            # Enterprise compliance check
            if data_quality.get('has_fallback', False):
                raise ValueError("❌ ENTERPRISE VIOLATION: Fallback data detected!")
            
            if data_quality.get('has_test_data', False):
                raise ValueError("❌ ENTERPRISE VIOLATION: Test data detected!")
            
            if data_quality.get('real_data_percentage', 0) < 100:
                raise ValueError("❌ ENTERPRISE VIOLATION: Not 100% real data!")
            
            self.logger.info(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
            
            return {
                'success': True,
                'data': data,
                'data_quality': data_quality,
                'rows': len(data),
                'columns': len(data.columns)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _stage_2_data_preprocessing(self) -> Dict[str, Any]:
        """Stage 2: ประมวลผลและทำความสะอาดข้อมูล"""
        try:
            self.logger.info("🧹 Preprocessing and cleaning data...")
            
            # Get data from previous stage
            data = self.pipeline_results.get('data_loading', {}).get('data')
            if data is None:
                raise ValueError("No data available from previous stage")
            
            # Elliott Wave pattern detection
            data_processed = self.data_processor.detect_elliott_wave_patterns(data)
            
            self.logger.info("✅ Data preprocessing completed")
            
            return {
                'success': True,
                'data_processed': data_processed,
                'elliott_wave_patterns_detected': True,
                'processing_steps': [
                    'Elliott Wave Pattern Detection',
                    'Price Swing Analysis',
                    'Fibonacci Retracement Calculation',
                    'Wave Relationship Analysis'
                ]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ==================== ENTERPRISE PROTECTION STAGES ====================
    
    def _stage_2b_enterprise_protection_analysis(self) -> Dict[str, Any]:
        """Stage 2b: Enterprise ML Protection Analysis"""
        try:
            self.logger.info("🛡️ Starting Enterprise ML Protection Analysis...")
            
            if not self.ml_protection:
                self.logger.warning("⚠️ ML Protection System not available, skipping...")
                return {
                    'success': True,
                    'stage': 'enterprise_protection_analysis',
                    'status': 'skipped',
                    'message': 'ML Protection System not initialized'
                }
            
            # Get preprocessed data from previous stage
            data_results = self.pipeline_results.get('data_preprocessing', {})
            features_df = data_results.get('processed_data')
            
            if features_df is None or features_df.empty:
                return {
                    'success': False,
                    'stage': 'enterprise_protection_analysis',
                    'error': 'No processed data available for protection analysis'
                }
            
            # Prepare target variable (simple binary classification for analysis)
            # This is just for protection analysis, actual targets will be prepared later
            if 'close' in features_df.columns:
                # Create a simple price direction target for analysis
                features_df = features_df.copy()
                features_df['future_close'] = features_df['close'].shift(-1)
                features_df['temp_target'] = (features_df['future_close'] > features_df['close']).astype(int)
                features_df = features_df.dropna()
                
                if len(features_df) < 100:
                    return {
                        'success': False,
                        'stage': 'enterprise_protection_analysis',
                        'error': 'Insufficient data for protection analysis'
                    }
                
                # Separate features and target for analysis
                analysis_features = features_df.select_dtypes(include=['number']).drop(columns=['future_close', 'temp_target'], errors='ignore')
                analysis_target = features_df['temp_target']
                
                # Run comprehensive protection analysis
                self.logger.info("🔍 Running comprehensive protection analysis...")
                protection_results = self.ml_protection.comprehensive_protection_analysis(
                    X=analysis_features,
                    y=analysis_target,
                    model=None,  # Will use default RandomForest
                    datetime_col='date' if 'date' in features_df.columns else None
                )
                
                # Check if analysis passed enterprise standards
                overall_assessment = protection_results.get('overall_assessment', {})
                enterprise_ready = overall_assessment.get('enterprise_ready', False)
                protection_status = overall_assessment.get('protection_status', 'UNKNOWN')
                risk_level = overall_assessment.get('risk_level', 'HIGH')
                
                # Log critical alerts
                alerts = protection_results.get('alerts', [])
                for alert in alerts:
                    self.logger.warning(alert)
                
                # Store results for later stages
                self.pipeline_results['initial_protection_analysis'] = protection_results
                
                return {
                    'success': True,
                    'stage': 'enterprise_protection_analysis',
                    'protection_results': protection_results,
                    'enterprise_ready': enterprise_ready,
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'alerts_count': len(alerts),
                    'recommendations_count': len(protection_results.get('recommendations', [])),
                    'message': f"Protection analysis completed - Status: {protection_status}, Risk: {risk_level}"
                }
            
            else:
                return {
                    'success': False,
                    'stage': 'enterprise_protection_analysis',
                    'error': 'Required price data not found for protection analysis'
                }
            
        except Exception as e:
            error_msg = f"Enterprise protection analysis failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'stage': 'enterprise_protection_analysis',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _stage_4b_pre_training_validation(self) -> Dict[str, Any]:
        """Stage 4b: Pre-Training Protection Validation"""
        try:
            self.logger.info("✅ Pre-Training Protection Validation...")
            
            if not self.ml_protection:
                return {
                    'success': True,
                    'stage': 'pre_training_validation',
                    'status': 'skipped',
                    'message': 'ML Protection System not available'
                }
            
            # Get feature selection results
            feature_results = self.pipeline_results.get('feature_selection', {})
            selected_features = feature_results.get('selected_features', [])
            training_data = feature_results.get('prepared_data')
            
            if training_data is None or len(selected_features) == 0:
                return {
                    'success': False,
                    'stage': 'pre_training_validation',
                    'error': 'No selected features or training data available'
                }
            
            # Validate selected features for protection issues
            self.logger.info("🔍 Validating selected features for protection issues...")
            
            # Check initial protection results for feature-specific issues
            initial_protection = self.pipeline_results.get('initial_protection_analysis', {})
            if initial_protection:
                leakage_data = initial_protection.get('data_leakage', {})
                suspicious_features = leakage_data.get('suspicious_features', [])
                future_features = leakage_data.get('future_features', [])
                
                # Check if any selected features are suspicious
                suspicious_selected = [f for f in selected_features if f in suspicious_features or f in future_features]
                
                if suspicious_selected:
                    self.logger.warning(f"⚠️ Suspicious features detected in selection: {suspicious_selected}")
                    self.pipeline_state['warnings'].append(f"Selected features contain potentially leaky features: {suspicious_selected}")
                
                # Feature quality assessment
                noise_data = initial_protection.get('noise_analysis', {})
                irrelevant_features = noise_data.get('feature_relevance', {}).get('irrelevant_features', [])
                irrelevant_selected = [f for f in selected_features if f in irrelevant_features]
                
                if irrelevant_selected:
                    self.logger.warning(f"⚠️ Irrelevant features detected in selection: {irrelevant_selected}")
                
                validation_results = {
                    'suspicious_features_count': len(suspicious_selected),
                    'irrelevant_features_count': len(irrelevant_selected),
                    'clean_features_count': len(selected_features) - len(suspicious_selected) - len(irrelevant_selected),
                    'feature_quality_score': (len(selected_features) - len(suspicious_selected) - len(irrelevant_selected)) / max(len(selected_features), 1),
                    'pre_training_approval': len(suspicious_selected) == 0,  # No suspicious features allowed
                    'warnings': []
                }
                
                if suspicious_selected:
                    validation_results['warnings'].append(f"Suspicious features detected: {suspicious_selected}")
                if irrelevant_selected:
                    validation_results['warnings'].append(f"Irrelevant features detected: {irrelevant_selected}")
                
                return {
                    'success': True,
                    'stage': 'pre_training_validation',
                    'validation_results': validation_results,
                    'approved_for_training': validation_results['pre_training_approval'],
                    'message': f"Pre-training validation completed - Quality Score: {validation_results['feature_quality_score']:.3f}"
                }
            
            # If no initial protection results, do basic validation
            return {
                'success': True,
                'stage': 'pre_training_validation',
                'status': 'basic_validation',
                'approved_for_training': True,
                'message': 'Basic pre-training validation completed'
            }
            
        except Exception as e:
            error_msg = f"Pre-training validation failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'stage': 'pre_training_validation',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _stage_6b_post_training_protection(self) -> Dict[str, Any]:
        """Stage 6b: Post-Training Protection Analysis"""
        try:
            self.logger.info("🛡️ Post-Training Protection Analysis...")
            
            if not self.ml_protection:
                return {
                    'success': True,
                    'stage': 'post_training_protection',
                    'status': 'skipped',
                    'message': 'ML Protection System not available'
                }
            
            # Get training results
            cnn_lstm_results = self.pipeline_results.get('cnn_lstm_training', {})
            dqn_results = self.pipeline_results.get('dqn_training', {})
            
            # Analyze training performance for overfitting
            post_training_analysis = {
                'overfitting_analysis': {},
                'model_performance': {},
                'training_stability': {},
                'enterprise_compliance': {}
            }
            
            # CNN-LSTM Analysis
            if cnn_lstm_results.get('success'):
                cnn_performance = cnn_lstm_results.get('performance_metrics', {})
                train_auc = cnn_performance.get('auc_score', 0)
                
                # Simple overfitting check (real implementation would be more sophisticated)
                if 'validation_auc' in cnn_performance:
                    val_auc = cnn_performance['validation_auc']
                    auc_gap = train_auc - val_auc
                    
                    post_training_analysis['overfitting_analysis']['cnn_lstm'] = {
                        'train_auc': train_auc,
                        'validation_auc': val_auc,
                        'performance_gap': auc_gap,
                        'overfitting_detected': auc_gap > 0.15,
                        'severity': 'HIGH' if auc_gap > 0.2 else 'MEDIUM' if auc_gap > 0.1 else 'LOW'
                    }
                
                post_training_analysis['model_performance']['cnn_lstm'] = {
                    'auc_score': train_auc,
                    'enterprise_ready': train_auc >= 0.70,
                    'performance_grade': 'A' if train_auc >= 0.80 else 'B' if train_auc >= 0.70 else 'C' if train_auc >= 0.60 else 'F'
                }
            
            # DQN Analysis
            if dqn_results.get('success'):
                dqn_performance = dqn_results.get('performance_metrics', {})
                total_reward = dqn_performance.get('total_reward', 0)
                
                post_training_analysis['model_performance']['dqn'] = {
                    'total_reward': total_reward,
                    'enterprise_ready': total_reward > 0,  # Simple check
                    'performance_grade': 'A' if total_reward > 1000 else 'B' if total_reward > 500 else 'C' if total_reward > 0 else 'F'
                }
            
            # Overall enterprise compliance
            cnn_ready = post_training_analysis['model_performance'].get('cnn_lstm', {}).get('enterprise_ready', False)
            dqn_ready = post_training_analysis['model_performance'].get('dqn', {}).get('enterprise_ready', False)
            no_overfitting = not any(
                analysis.get('overfitting_detected', False) 
                for analysis in post_training_analysis['overfitting_analysis'].values()
            )
            
            post_training_analysis['enterprise_compliance'] = {
                'cnn_lstm_ready': cnn_ready,
                'dqn_ready': dqn_ready,
                'no_overfitting': no_overfitting,
                'overall_ready': cnn_ready and dqn_ready and no_overfitting,
                'compliance_score': sum([cnn_ready, dqn_ready, no_overfitting]) / 3.0
            }
            
            return {
                'success': True,
                'stage': 'post_training_protection',
                'analysis_results': post_training_analysis,
                'enterprise_ready': post_training_analysis['enterprise_compliance']['overall_ready'],
                'compliance_score': post_training_analysis['enterprise_compliance']['compliance_score'],
                'message': f"Post-training analysis completed - Compliance: {post_training_analysis['enterprise_compliance']['compliance_score']:.3f}"
            }
            
        except Exception as e:
            error_msg = f"Post-training protection analysis failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'stage': 'post_training_protection',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _stage_8b_final_protection_report(self) -> Dict[str, Any]:
        """Stage 8b: Generate Final Protection Report"""
        try:
            self.logger.info("📋 Generating Final Enterprise Protection Report...")
            
            if not self.ml_protection:
                return {
                    'success': True,
                    'stage': 'final_protection_report',
                    'status': 'skipped',
                    'message': 'ML Protection System not available'
                }
            
            # Compile all protection results
            initial_protection = self.pipeline_results.get('initial_protection_analysis', {})
            pre_training_validation = self.pipeline_results.get('pre_training_validation', {})
            post_training_protection = self.pipeline_results.get('post_training_protection', {})
            
            # Generate comprehensive report
            report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"elliott_wave_protection_report_{report_timestamp}.txt"
            report_path = os.path.join(self.config.get('output_path', './results'), report_filename)
            
            # Generate report content
            if hasattr(self.ml_protection, 'generate_protection_report'):
                report_content = self.ml_protection.generate_protection_report(report_path)
            else:
                report_content = self._generate_simple_protection_report(
                    initial_protection, pre_training_validation, post_training_protection
                )
                
                # Save simple report
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
            
            # Final enterprise readiness assessment
            final_assessment = self._compute_final_enterprise_readiness(
                initial_protection, pre_training_validation, post_training_protection
            )
            
            self.logger.info(f"📄 Final protection report generated: {report_path}")
            
            return {
                'success': True,
                'stage': 'final_protection_report',
                'report_path': report_path,
                'report_content_preview': report_content[:500] + "..." if len(report_content) > 500 else report_content,
                'final_assessment': final_assessment,
                'enterprise_ready': final_assessment.get('enterprise_ready', False),
                'message': f"Final protection report generated - Enterprise Ready: {final_assessment.get('enterprise_ready', False)}"
            }
            
        except Exception as e:
            error_msg = f"Final protection report generation failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'stage': 'final_protection_report',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _generate_simple_protection_report(self, initial_protection: Dict, pre_training: Dict, post_training: Dict) -> str:
        """Generate a simple protection report if advanced system not available"""
        lines = []
        lines.append("🛡️ ELLIOTT WAVE ENTERPRISE PROTECTION REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Initial Protection Summary
        if initial_protection:
            overall = initial_protection.get('overall_assessment', {})
            lines.append("📊 INITIAL PROTECTION ANALYSIS")
            lines.append(f"Status: {overall.get('protection_status', 'UNKNOWN')}")
            lines.append(f"Risk Level: {overall.get('risk_level', 'UNKNOWN')}")
            lines.append(f"Enterprise Ready: {overall.get('enterprise_ready', False)}")
            lines.append("")
        
        # Pre-Training Validation
        if pre_training:
            validation = pre_training.get('validation_results', {})
            lines.append("✅ PRE-TRAINING VALIDATION")
            lines.append(f"Approved for Training: {pre_training.get('approved_for_training', False)}")
            lines.append(f"Feature Quality Score: {validation.get('feature_quality_score', 0):.3f}")
            lines.append("")
        
        # Post-Training Analysis
        if post_training:
            compliance = post_training.get('analysis_results', {}).get('enterprise_compliance', {})
            lines.append("🏆 POST-TRAINING ANALYSIS")
            lines.append(f"Overall Ready: {compliance.get('overall_ready', False)}")
            lines.append(f"Compliance Score: {compliance.get('compliance_score', 0):.3f}")
            lines.append("")
        
        lines.append("=" * 50)
        lines.append("Report End")
        
        return "\n".join(lines)
    
    def _compute_final_enterprise_readiness(self, initial_protection: Dict, pre_training: Dict, post_training: Dict) -> Dict:
        """Compute final enterprise readiness assessment"""
        try:
            readiness_factors = []
            
            # Initial protection readiness
            if initial_protection:
                initial_ready = initial_protection.get('overall_assessment', {}).get('enterprise_ready', False)
                readiness_factors.append(('initial_protection', initial_ready, 0.3))
            
            # Pre-training validation readiness
            if pre_training:
                pre_ready = pre_training.get('approved_for_training', False)
                readiness_factors.append(('pre_training', pre_ready, 0.2))
            
            # Post-training readiness
            if post_training:
                post_ready = post_training.get('enterprise_ready', False)
                readiness_factors.append(('post_training', post_ready, 0.5))
            
            # Calculate weighted score
            total_weight = sum(weight for _, _, weight in readiness_factors)
            if total_weight > 0:
                weighted_score = sum(ready * weight for _, ready, weight in readiness_factors) / total_weight
                overall_ready = weighted_score >= 0.8  # 80% threshold
            else:
                weighted_score = 0.0
                overall_ready = False
            
            return {
                'enterprise_ready': overall_ready,
                'readiness_score': weighted_score,
                'readiness_factors': {name: ready for name, ready, _ in readiness_factors},
                'assessment': 'ENTERPRISE_READY' if overall_ready else 'NEEDS_IMPROVEMENT',
                'recommendation': 'System approved for production use' if overall_ready else 'System requires improvements before production use'
            }
            
        except Exception as e:
            return {
                'enterprise_ready': False,
                'readiness_score': 0.0,
                'error': str(e),
                'assessment': 'ERROR',
                'recommendation': 'Unable to assess enterprise readiness due to error'
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """ส่งสถานะ Pipeline ปัจจุบัน"""
        return {
            'current_stage': self.pipeline_state.get('stage', 'not_started'),
            'progress_percentage': self.pipeline_state.get('progress', 0),
            'start_time': self.pipeline_state.get('start_time'),
            'errors': self.pipeline_state.get('errors', []),
            'warnings': self.pipeline_state.get('warnings', []),
            'stages_completed': len([r for r in self.pipeline_results.values() if r.get('success', False)])
        }
    
    def save_pipeline_results(self, filepath: str):
        """บันทึกผลลัพธ์ Pipeline"""
        try:
            import json
            
            # Prepare data for JSON serialization
            results_data = {
                'pipeline_state': self.pipeline_state,
                'pipeline_results': self.pipeline_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            results_data = convert_datetime(results_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Pipeline results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save pipeline results: {str(e)}")
    
    def run_integrated_pipeline(self, data: pd.DataFrame, selected_features: List[str], 
                               cnn_lstm_results: Dict, dqn_results: Dict) -> Dict[str, Any]:
        """รัน Integrated Pipeline เพื่อรวมผลลัพธ์จากทุก components"""
        try:
            self.logger.info("🔗 Starting Integrated Pipeline...")
            
            integration_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'components': {
                    'data_processed': True,
                    'features_selected': len(selected_features),
                    'selected_features': selected_features,
                    'cnn_lstm_trained': bool(cnn_lstm_results.get('model')),
                    'dqn_trained': bool(dqn_results.get('agent')),
                },
                'performance': {
                    'cnn_lstm_auc': cnn_lstm_results.get('auc_score', 0.0),
                    'dqn_total_reward': dqn_results.get('total_reward', 0.0),
                    'data_quality_score': len(data) / max(len(data), 1000) * 100  # Simple quality metric
                },
                'models_saved': {
                    'cnn_lstm_path': cnn_lstm_results.get('model_path', ''),
                    'dqn_agent_path': dqn_results.get('agent_path', '')
                },
                'integration_status': 'completed'
            }
            
            self.logger.info("✅ Integrated Pipeline completed successfully")
            return integration_results
            
        except Exception as e:
            self.logger.error(f"❌ Integrated Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
