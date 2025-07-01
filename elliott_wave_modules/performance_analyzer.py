#!/usr/bin/env python3
"""
📊 ELLIOTT WAVE PERFORMANCE ANALYZER
ตัววิเคราะห์ประสิทธิภาพสำหรับระบบ Elliott Wave

Enterprise Features:
- Comprehensive Performance Analysis
- Trading Metrics Calculation
- Risk Assessment
- Enterprise Reporting
- Production Monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

class ElliottWavePerformanceAnalyzer:
    """ตัววิเคราะห์ประสิทธิภาพ Elliott Wave ระดับ Enterprise"""
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance thresholds (Enterprise standards)
        self.min_auc = self.config.get('performance', {}).get('min_auc', 0.70)
        self.min_sharpe_ratio = self.config.get('performance', {}).get('min_sharpe_ratio', 1.5)
        self.max_drawdown = self.config.get('performance', {}).get('max_drawdown', 0.15)
        self.min_win_rate = self.config.get('performance', {}).get('min_win_rate', 0.60)
    
    def analyze_results(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """วิเคราะห์ผลลัพธ์จาก Pipeline"""
        try:
            self.logger.info("📊 Analyzing Elliott Wave system performance...")
            
            # Extract results from different stages
            cnn_lstm_results = pipeline_results.get('cnn_lstm_training', {}).get('cnn_lstm_results', {})
            dqn_results = pipeline_results.get('dqn_training', {}).get('dqn_results', {})
            feature_results = pipeline_results.get('feature_selection', {}).get('selection_results', {})
            
            # Analyze each component
            ml_analysis = self._analyze_ml_performance(cnn_lstm_results, feature_results)
            trading_analysis = self._analyze_trading_performance(dqn_results)
            risk_analysis = self._analyze_risk_metrics(pipeline_results)
            overall_analysis = self._analyze_overall_performance(ml_analysis, trading_analysis, risk_analysis)
            
            # Compile comprehensive analysis
            performance_analysis = {
                'ml_performance': ml_analysis,
                'trading_performance': trading_analysis,
                'risk_analysis': risk_analysis,
                'overall_performance': overall_analysis,
                'enterprise_grade': self._assess_enterprise_grade(overall_analysis),
                'recommendations': self._generate_recommendations(overall_analysis),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Performance analysis completed")
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Performance analysis failed: {str(e)}")
            return self._get_fallback_analysis()
    
    def _analyze_ml_performance(self, cnn_lstm_results: Dict, feature_results: Dict) -> Dict[str, Any]:
        """วิเคราะห์ประสิทธิภาพ ML Models"""
        try:
            # CNN-LSTM performance
            cnn_lstm_eval = cnn_lstm_results.get('evaluation_results', {})
            cnn_lstm_auc = cnn_lstm_eval.get('auc', 0.0)
            cnn_lstm_accuracy = cnn_lstm_eval.get('accuracy', 0.0)
            cnn_lstm_precision = cnn_lstm_eval.get('precision', 0.0)
            cnn_lstm_recall = cnn_lstm_eval.get('recall', 0.0)
            cnn_lstm_f1 = cnn_lstm_eval.get('f1_score', 0.0)
            
            # Feature selection performance
            feature_auc = feature_results.get('best_auc', 0.0)
            feature_count = feature_results.get('feature_count', 0)
            target_achieved = feature_results.get('target_achieved', False)
            
            # Calculate ML performance score
            ml_score = self._calculate_ml_score(cnn_lstm_auc, cnn_lstm_accuracy, feature_auc)
            
            return {
                'cnn_lstm_metrics': {
                    'auc': float(cnn_lstm_auc),
                    'accuracy': float(cnn_lstm_accuracy),
                    'precision': float(cnn_lstm_precision),
                    'recall': float(cnn_lstm_recall),
                    'f1_score': float(cnn_lstm_f1)
                },
                'feature_selection_metrics': {
                    'best_auc': float(feature_auc),
                    'selected_features': int(feature_count),
                    'target_achieved': bool(target_achieved)
                },
                'ml_performance_score': float(ml_score),
                'auc_target_met': bool(max(cnn_lstm_auc, feature_auc) >= self.min_auc),
                'model_quality': self._assess_model_quality(max(cnn_lstm_auc, feature_auc))
            }
            
        except Exception as e:
            self.logger.error(f"❌ ML performance analysis failed: {str(e)}")
            return {'ml_performance_score': 0.0, 'auc_target_met': False}
    
    def _analyze_trading_performance(self, dqn_results: Dict) -> Dict[str, Any]:
        """วิเคราะห์ประสิทธิภาพการเทรด"""
        try:
            # DQN trading performance
            dqn_eval = dqn_results.get('evaluation_results', {})
            
            total_return = dqn_eval.get('return_pct', 0.0)
            final_balance = dqn_eval.get('final_balance', 10000)
            total_trades = dqn_eval.get('total_trades', 0)
            
            # Calculate trading metrics
            sharpe_ratio = self._calculate_sharpe_ratio(total_return)
            win_rate = self._estimate_win_rate(total_return, total_trades)
            max_drawdown = self._estimate_max_drawdown(total_return)
            profit_factor = self._estimate_profit_factor(total_return)
            
            # Trading performance score
            trading_score = self._calculate_trading_score(sharpe_ratio, win_rate, max_drawdown)
            
            return {
                'trading_metrics': {
                    'total_return_pct': float(total_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'win_rate': float(win_rate),
                    'max_drawdown': float(max_drawdown),
                    'profit_factor': float(profit_factor),
                    'total_trades': int(total_trades),
                    'final_balance': float(final_balance)
                },
                'trading_performance_score': float(trading_score),
                'sharpe_target_met': bool(sharpe_ratio >= self.min_sharpe_ratio),
                'drawdown_acceptable': bool(max_drawdown <= self.max_drawdown),
                'win_rate_acceptable': bool(win_rate >= self.min_win_rate)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trading performance analysis failed: {str(e)}")
            return {'trading_performance_score': 0.0, 'sharpe_target_met': False}
    
    def _analyze_risk_metrics(self, pipeline_results: Dict) -> Dict[str, Any]:
        """วิเคราะห์ความเสี่ยง"""
        try:
            # Data quality risk
            data_loading = pipeline_results.get('data_loading', {})
            data_quality = data_loading.get('data_quality', {})
            
            data_risk_score = 100.0 if data_quality.get('real_data_percentage', 0) >= 100 else 50.0
            
            # Model risk
            validation_results = pipeline_results.get('quality_validation', {})
            model_risk_score = validation_results.get('quality_score', 0.0)
            
            # Overfitting risk
            cnn_lstm_results = pipeline_results.get('cnn_lstm_training', {}).get('cnn_lstm_results', {})
            training_results = cnn_lstm_results.get('training_results', {})
            
            train_acc = training_results.get('final_accuracy', 0.0)
            val_acc = training_results.get('final_val_accuracy', 0.0)
            overfitting_risk = abs(train_acc - val_acc) if train_acc > 0 and val_acc > 0 else 0.0
            
            # Overall risk score
            risk_score = (data_risk_score + model_risk_score) / 2
            
            return {
                'risk_metrics': {
                    'data_quality_risk': float(100 - data_risk_score),
                    'model_risk': float(100 - model_risk_score),
                    'overfitting_risk': float(overfitting_risk * 100),
                    'overall_risk_score': float(100 - risk_score)
                },
                'risk_assessment': self._assess_risk_level(risk_score),
                'risk_factors': self._identify_risk_factors(pipeline_results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Risk analysis failed: {str(e)}")
            return {'risk_metrics': {}, 'risk_assessment': 'High', 'risk_factors': []}
    
    def _analyze_overall_performance(self, ml_analysis: Dict, trading_analysis: Dict, risk_analysis: Dict) -> Dict[str, Any]:
        """วิเคราะห์ประสิทธิภาพโดยรวม"""
        try:
            # Component scores
            ml_score = ml_analysis.get('ml_performance_score', 0.0)
            trading_score = trading_analysis.get('trading_performance_score', 0.0)
            risk_score = 100 - risk_analysis.get('risk_metrics', {}).get('overall_risk_score', 100.0)
            
            # Weighted overall score
            overall_score = (ml_score * 0.4 + trading_score * 0.4 + risk_score * 0.2)
            
            # Performance grade
            performance_grade = self._calculate_performance_grade(overall_score)
            
            # Key metrics summary
            key_metrics = {
                'auc': ml_analysis.get('cnn_lstm_metrics', {}).get('auc', 0.0),
                'sharpe_ratio': trading_analysis.get('trading_metrics', {}).get('sharpe_ratio', 0.0),
                'max_drawdown': trading_analysis.get('trading_metrics', {}).get('max_drawdown', 0.0),
                'win_rate': trading_analysis.get('trading_metrics', {}).get('win_rate', 0.0) * 100,
                'total_return': trading_analysis.get('trading_metrics', {}).get('total_return_pct', 0.0)
            }
            
            return {
                'overall_score': float(overall_score),
                'performance_grade': performance_grade,
                'key_metrics': key_metrics,
                'component_scores': {
                    'ml_score': float(ml_score),
                    'trading_score': float(trading_score),
                    'risk_score': float(risk_score)
                },
                'enterprise_ready': bool(overall_score >= 70.0),
                'production_ready': bool(overall_score >= 80.0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Overall performance analysis failed: {str(e)}")
            return {'overall_score': 0.0, 'performance_grade': 'F', 'enterprise_ready': False}
    
    def _calculate_ml_score(self, cnn_lstm_auc: float, cnn_lstm_accuracy: float, feature_auc: float) -> float:
        """คำนวณคะแนน ML Performance"""
        best_auc = max(cnn_lstm_auc, feature_auc)
        
        # AUC score (70% weight)
        auc_score = min(best_auc / self.min_auc * 70, 70)
        
        # Accuracy score (30% weight)
        accuracy_score = min(cnn_lstm_accuracy * 30, 30)
        
        return auc_score + accuracy_score
    
    def _calculate_trading_score(self, sharpe_ratio: float, win_rate: float, max_drawdown: float) -> float:
        """คำนวณคะแนน Trading Performance"""
        # Sharpe ratio score (40% weight)
        sharpe_score = min(sharpe_ratio / self.min_sharpe_ratio * 40, 40)
        
        # Win rate score (30% weight)
        win_rate_score = min(win_rate / self.min_win_rate * 30, 30)
        
        # Drawdown score (30% weight)
        drawdown_score = max(30 - (max_drawdown / self.max_drawdown * 30), 0)
        
        return sharpe_score + win_rate_score + drawdown_score
    
    def _calculate_sharpe_ratio(self, total_return: float) -> float:
        """คำนวณ Sharpe Ratio (แบบประมาณ)"""
        # Simplified Sharpe ratio calculation
        # Assume risk-free rate = 2%, volatility = 15%
        risk_free_rate = 0.02
        volatility = 0.15
        
        annual_return = total_return / 100  # Convert percentage to decimal
        sharpe = (annual_return - risk_free_rate) / volatility
        
        return max(sharpe, 0.0)
    
    def _estimate_win_rate(self, total_return: float, total_trades: int) -> float:
        """ประมาณ Win Rate"""
        if total_trades == 0:
            return 0.5
        
        # Simple estimation based on return
        if total_return > 10:
            return 0.70
        elif total_return > 5:
            return 0.65
        elif total_return > 0:
            return 0.60
        else:
            return 0.40
    
    def _estimate_max_drawdown(self, total_return: float) -> float:
        """ประมาณ Maximum Drawdown"""
        # Simple estimation: drawdown typically 1/3 of return volatility
        return abs(total_return) * 0.3 / 100
    
    def _estimate_profit_factor(self, total_return: float) -> float:
        """ประมาณ Profit Factor"""
        if total_return > 10:
            return 1.8
        elif total_return > 5:
            return 1.5
        elif total_return > 0:
            return 1.2
        else:
            return 0.8
    
    def _assess_model_quality(self, auc: float) -> str:
        """ประเมินคุณภาพโมเดล"""
        if auc >= 0.85:
            return "Excellent"
        elif auc >= 0.75:
            return "Very Good"
        elif auc >= 0.70:
            return "Good"
        elif auc >= 0.60:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_risk_level(self, risk_score: float) -> str:
        """ประเมินระดับความเสี่ยง"""
        if risk_score >= 80:
            return "Low"
        elif risk_score >= 60:
            return "Medium"
        else:
            return "High"
    
    def _identify_risk_factors(self, pipeline_results: Dict) -> List[str]:
        """ระบุปัจจัยเสี่ยง"""
        risk_factors = []
        
        # Check AUC
        feature_results = pipeline_results.get('feature_selection', {}).get('selection_results', {})
        if feature_results.get('best_auc', 0.0) < self.min_auc:
            risk_factors.append(f"AUC below target ({self.min_auc})")
        
        # Check overfitting
        cnn_lstm_results = pipeline_results.get('cnn_lstm_training', {}).get('cnn_lstm_results', {})
        training_results = cnn_lstm_results.get('training_results', {})
        
        train_acc = training_results.get('final_accuracy', 0.0)
        val_acc = training_results.get('final_val_accuracy', 0.0)
        
        if abs(train_acc - val_acc) > 0.1:
            risk_factors.append("Potential overfitting detected")
        
        # Check data quality
        data_loading = pipeline_results.get('data_loading', {})
        if data_loading.get('rows', 0) < 1000:
            risk_factors.append("Limited training data")
        
        return risk_factors
    
    def _calculate_performance_grade(self, overall_score: float) -> str:
        """คำนวณเกรดประสิทธิภาพ"""
        if overall_score >= 90:
            return "A+"
        elif overall_score >= 85:
            return "A"
        elif overall_score >= 80:
            return "A-"
        elif overall_score >= 75:
            return "B+"
        elif overall_score >= 70:
            return "B"
        elif overall_score >= 65:
            return "B-"
        elif overall_score >= 60:
            return "C+"
        elif overall_score >= 55:
            return "C"
        elif overall_score >= 50:
            return "C-"
        else:
            return "F"
    
    def _assess_enterprise_grade(self, overall_analysis: Dict) -> Dict[str, Any]:
        """ประเมินมาตรฐาน Enterprise"""
        overall_score = overall_analysis.get('overall_score', 0.0)
        
        enterprise_criteria = {
            'performance_threshold': overall_score >= 70.0,
            'production_ready': overall_score >= 80.0,
            'auc_requirement': overall_analysis.get('key_metrics', {}).get('auc', 0.0) >= self.min_auc,
            'risk_acceptable': overall_analysis.get('component_scores', {}).get('risk_score', 0.0) >= 60.0,
            'enterprise_grade': overall_score >= 75.0
        }
        
        enterprise_score = sum(enterprise_criteria.values()) / len(enterprise_criteria) * 100
        
        return {
            'enterprise_criteria': enterprise_criteria,
            'enterprise_score': float(enterprise_score),
            'enterprise_ready': bool(enterprise_score >= 80.0),
            'certification_level': self._get_certification_level(enterprise_score)
        }
    
    def _get_certification_level(self, enterprise_score: float) -> str:
        """ระดับการรับรอง Enterprise"""
        if enterprise_score >= 95:
            return "Enterprise Gold"
        elif enterprise_score >= 85:
            return "Enterprise Silver"
        elif enterprise_score >= 75:
            return "Enterprise Bronze"
        elif enterprise_score >= 60:
            return "Production Ready"
        else:
            return "Development Grade"
    
    def _generate_recommendations(self, overall_analysis: Dict) -> List[str]:
        """สร้างคำแนะนำการปรับปรุง"""
        recommendations = []
        
        overall_score = overall_analysis.get('overall_score', 0.0)
        key_metrics = overall_analysis.get('key_metrics', {})
        
        # AUC recommendations
        if key_metrics.get('auc', 0.0) < self.min_auc:
            recommendations.append("Improve feature engineering and model architecture to achieve AUC ≥ 70%")
        
        # Sharpe ratio recommendations
        if key_metrics.get('sharpe_ratio', 0.0) < self.min_sharpe_ratio:
            recommendations.append("Optimize trading strategy to improve risk-adjusted returns")
        
        # Drawdown recommendations
        if key_metrics.get('max_drawdown', 0.0) > self.max_drawdown:
            recommendations.append("Implement better risk management to reduce maximum drawdown")
        
        # Win rate recommendations
        if key_metrics.get('win_rate', 0.0) < (self.min_win_rate * 100):
            recommendations.append("Refine entry and exit signals to improve win rate")
        
        # Overall score recommendations
        if overall_score < 70:
            recommendations.append("System requires significant improvements before enterprise deployment")
        elif overall_score < 80:
            recommendations.append("System shows good potential but needs optimization for production use")
        else:
            recommendations.append("System meets enterprise standards and is ready for production deployment")
        
        return recommendations
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback Analysis เมื่อเกิดข้อผิดพลาด"""
        return {
            'ml_performance': {'ml_performance_score': 0.0, 'auc_target_met': False},
            'trading_performance': {'trading_performance_score': 0.0, 'sharpe_target_met': False},
            'risk_analysis': {'risk_assessment': 'High', 'risk_factors': ['Analysis failed']},
            'overall_performance': {
                'overall_score': 0.0,
                'performance_grade': 'F',
                'enterprise_ready': False,
                'production_ready': False
            },
            'enterprise_grade': {
                'enterprise_ready': False,
                'certification_level': 'Development Grade'
            },
            'recommendations': ['System analysis failed - manual review required'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_performance_report(self, analysis_results: Dict[str, Any]) -> str:
        """สร้างรายงานประสิทธิภาพ"""
        try:
            overall = analysis_results.get('overall_performance', {})
            key_metrics = overall.get('key_metrics', {})
            enterprise = analysis_results.get('enterprise_grade', {})
            
            report = f"""
🏢 NICEGOLD ELLIOTT WAVE PERFORMANCE REPORT
{'='*60}

📊 OVERALL PERFORMANCE
  Score: {overall.get('overall_score', 0.0):.1f}/100
  Grade: {overall.get('performance_grade', 'F')}
  Enterprise Ready: {'✅' if enterprise.get('enterprise_ready', False) else '❌'}
  Certification: {enterprise.get('certification_level', 'N/A')}

🎯 KEY METRICS
  AUC Score: {key_metrics.get('auc', 0.0):.3f}
  Sharpe Ratio: {key_metrics.get('sharpe_ratio', 0.0):.2f}
  Max Drawdown: {key_metrics.get('max_drawdown', 0.0):.1%}
  Win Rate: {key_metrics.get('win_rate', 0.0):.1f}%
  Total Return: {key_metrics.get('total_return', 0.0):.1f}%

📋 RECOMMENDATIONS
"""
            
            recommendations = analysis_results.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                report += f"  {i}. {rec}\n"
            
            return report
            
        except Exception as e:
            return f"❌ Failed to generate performance report: {str(e)}"
    
    def export_analysis_to_dict(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """ส่งออกการวิเคราะห์เป็น Dictionary"""
        return {
            'performance_analysis': analysis_results,
            'export_timestamp': datetime.now().isoformat(),
            'analyzer_version': '2.0 DIVINE EDITION'
        }
