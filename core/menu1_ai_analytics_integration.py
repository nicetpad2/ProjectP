#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 MENU 1 AI ANALYTICS INTEGRATION
==================================

Integration layer for Advanced AI Analytics with Menu 1 Elliott Wave System
Real-time analytics, pattern detection, and performance monitoring

🎯 Key Features:
- Seamless integration with Menu 1
- Real-time analytics during trading
- Performance monitoring and optimization
- Advanced pattern detection insights
- Market sentiment analysis
- AI-powered trading recommendations

📊 Integration Components:
- Analytics Pipeline Manager
- Real-time Data Processing
- Performance Metrics Dashboard
- Pattern Detection Integration
- Market Sentiment Integration
- Trading Signal Enhancement
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

# Data processing
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Project imports
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

# Advanced AI Analytics
try:
    from core.advanced_ai_analytics import (
        AdvancedAIAnalytics, NeuralPerformanceMonitor,
        AdvancedPatternDetector, MarketPredictor
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    print("⚠️ Advanced AI Analytics not available")

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Menu1AIAnalyticsIntegration:
    """
    🚀 Menu 1 AI Analytics Integration Manager

    Manages the integration between Menu 1 Elliott Wave System and
    Advanced AI Analytics
    """

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.logger = get_unified_logger()

        # Initialize analytics components
        if ANALYTICS_AVAILABLE:
            self.analytics = AdvancedAIAnalytics()
            self.performance_monitor = NeuralPerformanceMonitor()
            self.pattern_detector = AdvancedPatternDetector()
            self.market_predictor = MarketPredictor()
        else:
            self.analytics = None
            self.performance_monitor = None
            self.pattern_detector = None
            self.market_predictor = None

        # Analytics state
        self.analytics_enabled = ANALYTICS_AVAILABLE
        self.real_time_monitoring = False
        self.analytics_cache = {}
        self.performance_history = []

        # Initialize logger
        self.logger.info("Menu 1 AI Analytics Integration initialized")

    def initialize_analytics(self, enable_real_time: bool = True) -> bool:
        """Initialize analytics system for Menu 1 integration"""

        try:
            if not self.analytics_enabled:
                self.logger.warning(
                    "Analytics not available - running in basic mode"
                )
                return False

            self.real_time_monitoring = enable_real_time

            if self.console:
                self.console.print(Panel.fit(
                    "🚀 Initializing AI Analytics Integration",
                    style="bold blue"
                ))

            # Initialize analytics components
            self.logger.info("Initializing analytics components...")

            # Test analytics functionality
            test_data = self._generate_test_data()
            test_results = self.analytics.run_comprehensive_analysis(test_data)

            if 'error' not in test_results:
                self.logger.success("Analytics initialization successful")
                if self.console:
                    self.console.print("✅ Analytics system ready")
                return True
            else:
                self.logger.error(
                    f"Analytics initialization failed: {test_results['error']}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error initializing analytics: {e}")
            return False
    
    def analyze_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data with advanced AI analytics
        
        Args:
            market_data: DataFrame with market data (OHLCV)
            
        Returns:
            Dictionary with analysis results
        """
        
        try:
            if not self.analytics_enabled:
                return {'error': 'Analytics not available'}
            
            # Log analysis start
            self.logger.info(f"Starting market analysis for {len(market_data)} data points")
            
            # Run comprehensive analysis
            if self.console:
                with self.console.status("🔍 Analyzing market data..."):
                    analysis_results = self.analytics.run_comprehensive_analysis(market_data)
            else:
                analysis_results = self.analytics.run_comprehensive_analysis(market_data)
            
            # Cache results
            self.analytics_cache['latest_analysis'] = analysis_results
            self.analytics_cache['timestamp'] = datetime.now().isoformat()
            
            # Log results
            if 'error' not in analysis_results:
                insights_count = len(analysis_results.get('insights', []))
                recommendations_count = len(analysis_results.get('recommendations', []))
                
                self.logger.success(f"Analysis completed: {insights_count} insights, {recommendations_count} recommendations")
            else:
                self.logger.error(f"Analysis failed: {analysis_results['error']}")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {e}")
            return {'error': str(e)}
    
    def monitor_model_performance(self, predictions: np.ndarray, actual_values: np.ndarray, 
                                model_name: str = "Elliott-Wave-CNN-LSTM") -> Dict[str, Any]:
        """
        Monitor model performance during trading
        
        Args:
            predictions: Model predictions
            actual_values: Actual market values
            model_name: Name of the model
            
        Returns:
            Performance metrics
        """
        
        try:
            if not self.analytics_enabled:
                return {'error': 'Analytics not available'}
            
            # Log performance monitoring
            self.logger.info(f"Monitoring performance for {model_name}")
            
            # Run performance analysis
            performance_results = self.performance_monitor.monitor_model_performance(
                predictions, actual_values, model_name
            )
            
            # Store performance history
            self.performance_history.append(performance_results)
            
            # Log performance
            if 'error' not in performance_results:
                health_score = performance_results.get('health_score', 0)
                self.logger.info(f"Model health score: {health_score:.2f}%")
                
                # Alert if performance is poor
                if health_score < 60:
                    self.logger.warning(f"Model performance below threshold: {health_score:.2f}%")
                elif health_score > 80:
                    self.logger.success(f"Excellent model performance: {health_score:.2f}%")
            
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")
            return {'error': str(e)}
    
    def detect_elliott_wave_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Elliott Wave patterns with enhanced AI analysis
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Pattern detection results
        """
        
        try:
            if not self.analytics_enabled:
                return {'error': 'Analytics not available'}
            
            # Log pattern detection
            self.logger.info("Detecting Elliott Wave patterns")
            
            # Run pattern detection
            pattern_results = self.pattern_detector.detect_elliott_wave_patterns(price_data)
            
            # Log pattern results
            if 'error' not in pattern_results:
                patterns_count = pattern_results.get('patterns_detected', 0)
                confidence = pattern_results.get('confidence_score', 0)
                
                self.logger.info(f"Detected {patterns_count} patterns with {confidence:.2f}% confidence")
                
                # Alert for high-confidence patterns
                if confidence > 75:
                    self.logger.success(f"High-confidence Elliott Wave pattern detected: {confidence:.2f}%")
            
            return pattern_results
            
        except Exception as e:
            self.logger.error(f"Error detecting Elliott Wave patterns: {e}")
            return {'error': str(e)}
    
    def predict_market_sentiment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict market sentiment for trading decisions
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Market sentiment prediction
        """
        
        try:
            if not self.analytics_enabled:
                return {'error': 'Analytics not available'}
            
            # Log sentiment prediction
            self.logger.info("Predicting market sentiment")
            
            # Run sentiment analysis
            sentiment_results = self.market_predictor.predict_market_sentiment(market_data)
            
            # Log sentiment results
            if 'error' not in sentiment_results:
                sentiment_label = sentiment_results.get('sentiment_label', 'Unknown')
                confidence = sentiment_results.get('confidence', 0)
                
                self.logger.info(f"Market sentiment: {sentiment_label} (Confidence: {confidence:.2f}%)")
                
                # Alert for extreme sentiment
                sentiment_score = sentiment_results.get('sentiment_score', 50)
                if sentiment_score > 80:
                    self.logger.warning("Very bullish sentiment detected")
                elif sentiment_score < 20:
                    self.logger.warning("Very bearish sentiment detected")
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error predicting market sentiment: {e}")
            return {'error': str(e)}
    
    def generate_trading_insights(self, market_data: pd.DataFrame, 
                                model_predictions: Optional[np.ndarray] = None,
                                actual_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive trading insights combining all analytics
        
        Args:
            market_data: DataFrame with market data
            model_predictions: Optional model predictions
            actual_values: Optional actual values
            
        Returns:
            Comprehensive trading insights
        """
        
        try:
            if not self.analytics_enabled:
                return {'error': 'Analytics not available'}
            
            self.logger.info("Generating comprehensive trading insights")
            
            # Initialize insights
            insights = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(market_data),
                'components': {}
            }
            
            # 1. Market Analysis
            if self.console:
                self.console.print("📊 Analyzing market data...")
            
            market_analysis = self.analyze_market_data(market_data)
            insights['components']['market_analysis'] = market_analysis
            
            # 2. Pattern Detection
            if self.console:
                self.console.print("🌊 Detecting Elliott Wave patterns...")
            
            pattern_analysis = self.detect_elliott_wave_patterns(market_data)
            insights['components']['pattern_analysis'] = pattern_analysis
            
            # 3. Sentiment Analysis
            if self.console:
                self.console.print("💭 Analyzing market sentiment...")
            
            sentiment_analysis = self.predict_market_sentiment(market_data)
            insights['components']['sentiment_analysis'] = sentiment_analysis
            
            # 4. Performance Monitoring (if model data available)
            if model_predictions is not None and actual_values is not None:
                if self.console:
                    self.console.print("⚡ Monitoring model performance...")
                
                performance_analysis = self.monitor_model_performance(
                    model_predictions, actual_values
                )
                insights['components']['performance_analysis'] = performance_analysis
            
            # 5. Generate combined insights
            combined_insights = self._generate_combined_insights(insights['components'])
            insights['insights'] = combined_insights
            
            # 6. Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(insights['components'])
            insights['recommendations'] = trading_recommendations
            
            # Log completion
            self.logger.success(f"Generated {len(combined_insights)} insights and {len(trading_recommendations)} recommendations")
            
            # Display summary
            if self.console:
                self._display_insights_summary(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating trading insights: {e}")
            return {'error': str(e)}
    
    def _generate_combined_insights(self, components: Dict[str, Any]) -> List[str]:
        """Generate combined insights from all analytics components"""
        
        insights = []
        
        # Market analysis insights
        if 'market_analysis' in components:
            market = components['market_analysis']
            if 'insights' in market:
                insights.extend(market['insights'])
        
        # Pattern analysis insights
        if 'pattern_analysis' in components:
            patterns = components['pattern_analysis']
            if 'patterns_detected' in patterns:
                pattern_count = patterns['patterns_detected']
                confidence = patterns.get('confidence_score', 0)
                
                if pattern_count > 0:
                    insights.append(f"🌊 {pattern_count} Elliott Wave patterns detected with {confidence:.1f}% confidence")
                    
                    if confidence > 70:
                        insights.append("✅ High-confidence patterns suggest clear market structure")
                    elif confidence < 40:
                        insights.append("⚠️ Low-confidence patterns suggest market uncertainty")
        
        # Sentiment analysis insights
        if 'sentiment_analysis' in components:
            sentiment = components['sentiment_analysis']
            if 'sentiment_label' in sentiment:
                sentiment_label = sentiment['sentiment_label']
                sentiment_score = sentiment.get('sentiment_score', 50)
                
                insights.append(f"💭 Market sentiment: {sentiment_label} ({sentiment_score:.1f}/100)")
                
                if sentiment_score > 75:
                    insights.append("🚀 Strong bullish sentiment supports upward momentum")
                elif sentiment_score < 25:
                    insights.append("🐻 Strong bearish sentiment suggests downward pressure")
        
        # Performance insights
        if 'performance_analysis' in components:
            performance = components['performance_analysis']
            if 'health_score' in performance:
                health_score = performance['health_score']
                
                if health_score > 80:
                    insights.append(f"⚡ Excellent model performance: {health_score:.1f}% health score")
                elif health_score < 60:
                    insights.append(f"⚠️ Model performance needs attention: {health_score:.1f}% health score")
        
        return insights
    
    def _generate_trading_recommendations(self, components: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analytics"""
        
        recommendations = []
        
        # Pattern-based recommendations
        if 'pattern_analysis' in components:
            patterns = components['pattern_analysis']
            confidence = patterns.get('confidence_score', 0)
            
            if confidence > 70:
                recommendations.append("🌊 High-confidence Elliott Wave patterns - consider following the trend")
            elif confidence < 40:
                recommendations.append("⚠️ Low pattern confidence - wait for clearer signals")
        
        # Sentiment-based recommendations
        if 'sentiment_analysis' in components:
            sentiment = components['sentiment_analysis']
            sentiment_score = sentiment.get('sentiment_score', 50)
            
            if sentiment_score > 75:
                recommendations.append("🚀 Strong bullish sentiment - consider long positions")
            elif sentiment_score < 25:
                recommendations.append("🐻 Strong bearish sentiment - consider short positions")
            else:
                recommendations.append("⚖️ Neutral sentiment - consider range-bound strategies")
        
        # Performance-based recommendations
        if 'performance_analysis' in components:
            performance = components['performance_analysis']
            health_score = performance.get('health_score', 0)
            
            if health_score < 60:
                recommendations.append("🔧 Model performance low - consider reducing position sizes")
            elif health_score > 80:
                recommendations.append("✅ Model performing well - suitable for normal position sizing")
        
        # Risk management recommendations
        recommendations.append("🛡️ Always use proper risk management and position sizing")
        recommendations.append("📊 Monitor analytics continuously for changing market conditions")
        
        return recommendations
    
    def _display_insights_summary(self, insights: Dict[str, Any]):
        """Display insights summary in beautiful format"""
        
        if not self.console:
            return
        
        # Create summary table
        table = Table(title="🚀 Trading Insights Summary")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Key Insight", style="magenta")
        
        components = insights['components']
        
        # Add component statuses
        if 'market_analysis' in components:
            table.add_row("Market Analysis", "✅ Completed", "Multi-component analysis")
        
        if 'pattern_analysis' in components:
            patterns = components['pattern_analysis']
            pattern_count = patterns.get('patterns_detected', 0)
            table.add_row("Pattern Detection", "✅ Completed", f"{pattern_count} patterns found")
        
        if 'sentiment_analysis' in components:
            sentiment = components['sentiment_analysis']
            sentiment_label = sentiment.get('sentiment_label', 'Unknown')
            table.add_row("Sentiment Analysis", "✅ Completed", f"Sentiment: {sentiment_label}")
        
        if 'performance_analysis' in components:
            performance = components['performance_analysis']
            health_score = performance.get('health_score', 0)
            table.add_row("Performance Monitor", "✅ Completed", f"Health: {health_score:.1f}%")
        
        self.console.print(table)
        
        # Display insights
        if insights.get('insights'):
            self.console.print("\n🧠 Key Insights:")
            for insight in insights['insights']:
                self.console.print(f"  • {insight}")
        
        # Display recommendations
        if insights.get('recommendations'):
            self.console.print("\n💡 Trading Recommendations:")
            for rec in insights['recommendations']:
                self.console.print(f"  • {rec}")
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate test data for analytics initialization"""
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Generate realistic price data
        price_base = 2000
        price_trend = np.cumsum(np.random.randn(100) * 0.1)
        close_prices = price_base + price_trend + np.random.randn(100) * 2
        
        # Generate OHLC data
        open_prices = close_prices + np.random.randn(100) * 0.5
        high_prices = np.maximum(open_prices, close_prices) + abs(np.random.randn(100) * 2)
        low_prices = np.minimum(open_prices, close_prices) - abs(np.random.randn(100) * 2)
        
        # Generate volume data
        volumes = np.random.randint(100, 1000, 100)
        
        return pd.DataFrame({
            'DateTime': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        })
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get current analytics system status"""
        
        status = {
            'analytics_enabled': self.analytics_enabled,
            'real_time_monitoring': self.real_time_monitoring,
            'components_available': {
                'advanced_analytics': self.analytics is not None,
                'performance_monitor': self.performance_monitor is not None,
                'pattern_detector': self.pattern_detector is not None,
                'market_predictor': self.market_predictor is not None
            },
            'performance_history_count': len(self.performance_history),
            'cache_available': bool(self.analytics_cache),
            'last_analysis': self.analytics_cache.get('timestamp', 'Never')
        }
        
        return status
    
    def export_analytics_report(self, output_path: Optional[str] = None) -> str:
        """Export comprehensive analytics report"""
        
        try:
            if not self.analytics_enabled:
                return "Analytics not available for report generation"
            
            # Generate report
            report = self.analytics.generate_analytics_report(output_path)
            
            self.logger.success("Analytics report generated successfully")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            return f"Error generating report: {e}"

def demo_menu1_analytics_integration():
    """Demo function to showcase Menu 1 analytics integration"""
    
    console = Console() if RICH_AVAILABLE else None
    if console:
        console.print(Panel.fit("🚀 Menu 1 AI Analytics Integration Demo", style="bold blue"))
    
    # Create integration system
    integration = Menu1AIAnalyticsIntegration()
    
    # Initialize analytics
    print("\n🔧 Initializing analytics system...")
    success = integration.initialize_analytics()
    
    if success:
        print("✅ Analytics system initialized successfully")
        
        # Generate sample market data
        print("\n📊 Generating sample market data...")
        market_data = integration._generate_test_data()
        
        # Generate comprehensive insights
        print("\n🧠 Generating comprehensive trading insights...")
        insights = integration.generate_trading_insights(market_data)
        
        # Display results
        if 'error' not in insights:
            print(f"\n✅ Generated {len(insights.get('insights', []))} insights")
            print(f"📝 Generated {len(insights.get('recommendations', []))} recommendations")
        
        # Show analytics status
        print("\n📊 Analytics Status:")
        status = integration.get_analytics_status()
        for key, value in status.items():
            print(f"  • {key}: {value}")
        
        print("\n🎯 Demo completed successfully!")
        
    else:
        print("❌ Failed to initialize analytics system")
    
    return integration

if __name__ == "__main__":
    # Run demo
    demo_integration = demo_menu1_analytics_integration()
    print("\n🎉 Menu 1 AI Analytics Integration Demo completed!")
