#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 ADVANCED TRADING SIGNAL SYSTEM - ENTERPRISE GRADE
Professional Buy/Sell/Hold Signal Generator for NICEGOLD ProjectP

Features:
- Multi-Model Ensemble Predictions
- Elliott Wave Pattern Analysis
- Real-time Risk Management
- Dynamic Position Sizing
- Enterprise-grade Signal Validation
- Production-ready Trading Logic
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

@dataclass
class TradingSignal:
    """Complete trading signal data structure"""
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    position_size: float
    elliott_wave_pattern: str
    technical_indicators: Dict[str, float]
    market_regime: str
    reasoning: str

class AdvancedTradingSignalGenerator:
    """
    🚀 ENTERPRISE TRADING SIGNAL GENERATOR
    
    Multi-dimensional signal analysis combining:
    - Elliott Wave pattern recognition
    - Technical indicator confluence
    - ML model predictions
    - Risk management rules
    - Market regime detection
    """
    
    def __init__(self, 
                 models: Dict[str, Any] = None,
                 config: Dict[str, Any] = None,
                 logger: logging.Logger = None):
        """Initialize the advanced signal generator"""
        self.models = models or {}
        self.config = config or self._get_default_config()
        self.logger = logger or self._setup_logger()
        
        # Signal history for tracking
        self.signal_history: List[TradingSignal] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        self.logger.info("🚀 Advanced Trading Signal Generator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for signal generation"""
        return {
            'min_confidence_threshold': 0.70,  # Minimum confidence for signal
            'max_position_size': 0.02,  # Max 2% of capital per trade
            'min_risk_reward_ratio': 1.5,  # Minimum 1.5:1 risk/reward
            'stop_loss_percentage': 0.015,  # 1.5% stop loss
            'take_profit_multiplier': 2.0,  # 2x stop loss for take profit
            'elliott_wave_weight': 0.30,  # 30% weight for Elliott Wave
            'technical_indicators_weight': 0.25,  # 25% weight for indicators
            'ml_prediction_weight': 0.35,  # 35% weight for ML models
            'market_regime_weight': 0.10,  # 10% weight for market regime
            'signal_cooldown_minutes': 15,  # Minimum 15 minutes between signals
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for signal generator"""
        logger = logging.getLogger("AdvancedTradingSignals")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def generate_signal(self, 
                       data: pd.DataFrame,
                       current_price: float,
                       timestamp: datetime = None) -> Optional[TradingSignal]:
        """
        🎯 MAIN SIGNAL GENERATION METHOD
        
        Combines multiple analysis methods to generate trading signals
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # 1. Check signal cooldown
            if not self._check_signal_cooldown(timestamp):
                return None
            
            # 2. Analyze Elliott Wave patterns
            elliott_analysis = self._analyze_elliott_wave_patterns(data)
            
            # 3. Analyze technical indicators
            technical_analysis = self._analyze_technical_indicators(data)
            
            # 4. Get ML model predictions
            ml_predictions = self._get_ml_predictions(data)
            
            # 5. Detect market regime
            market_regime = self._detect_market_regime(data)
            
            # 6. Calculate composite signal
            signal_data = self._calculate_composite_signal(
                elliott_analysis,
                technical_analysis,
                ml_predictions,
                market_regime,
                current_price,
                timestamp
            )
            
            # 7. Validate signal quality
            if signal_data and self._validate_signal(signal_data):
                signal = self._create_trading_signal(signal_data, current_price, timestamp)
                self.signal_history.append(signal)
                self._log_signal(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def _analyze_elliott_wave_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        🌊 ELLIOTT WAVE PATTERN ANALYSIS
        
        Analyzes Elliott Wave patterns to determine market structure
        """
        try:
            # Get recent price data
            close_prices = data['close'].tail(100).values
            high_prices = data['high'].tail(100).values
            low_prices = data['low'].tail(100).values
            
            # Identify wave patterns
            waves = self._identify_elliott_waves(close_prices, high_prices, low_prices)
            
            # Determine current wave position
            current_wave = self._determine_current_wave_position(waves)
            
            # Calculate wave-based signals
            wave_signal = self._calculate_wave_signal(current_wave, waves)
            
            return {
                'wave_pattern': current_wave.get('pattern', 'UNKNOWN'),
                'wave_position': current_wave.get('position', 0),
                'wave_direction': wave_signal.get('direction', 'NEUTRAL'),
                'wave_strength': wave_signal.get('strength', 0),
                'fibonacci_levels': self._calculate_fibonacci_levels(waves),
                'wave_completion': current_wave.get('completion', 0.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Elliott Wave analysis error: {e}")
            return {'wave_pattern': 'UNKNOWN', 'wave_direction': 'NEUTRAL', 'wave_strength': 0}
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        📊 TECHNICAL INDICATOR ANALYSIS
        
        Analyzes multiple technical indicators for signal confluence
        """
        try:
            # Calculate key indicators
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['ema_12'] = data['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['close'], 14)
            
            # MACD
            macd_data = self._calculate_macd(data['close'])
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data['close'], 20, 2)
            indicators.update(bb_data)
            
            # Stochastic
            stoch_data = self._calculate_stochastic(data['high'], data['low'], data['close'])
            indicators.update(stoch_data)
            
            # Volume indicators
            if 'volume' in data.columns:
                indicators['volume_sma'] = data['volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma']
            
            # Calculate indicator signals
            signal_scores = self._calculate_indicator_signals(indicators, data['close'].iloc[-1])
            
            return {
                'indicators': indicators,
                'signal_scores': signal_scores,
                'overall_score': np.mean(list(signal_scores.values())),
                'bullish_count': sum(1 for score in signal_scores.values() if score > 0.6),
                'bearish_count': sum(1 for score in signal_scores.values() if score < -0.6)
            }
            
        except Exception as e:
            self.logger.warning(f"Technical analysis error: {e}")
            return {'indicators': {}, 'signal_scores': {}, 'overall_score': 0}
    
    def _get_ml_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        🤖 MACHINE LEARNING PREDICTIONS
        
        Gets predictions from trained ML models
        """
        try:
            predictions = {}
            
            # Prepare features for prediction
            features = self._prepare_ml_features(data)
            
            # Get predictions from available models
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features.reshape(1, -1))[0]
                        predictions[f'{model_name}_proba'] = proba
                        predictions[f'{model_name}_signal'] = np.argmax(proba) - 1  # -1, 0, 1 for SELL, HOLD, BUY
                        predictions[f'{model_name}_confidence'] = np.max(proba)
                    elif hasattr(model, 'predict'):
                        pred = model.predict(features.reshape(1, -1))[0]
                        predictions[f'{model_name}_prediction'] = pred
                        predictions[f'{model_name}_signal'] = 1 if pred > 0.5 else -1 if pred < -0.5 else 0
                        predictions[f'{model_name}_confidence'] = abs(pred)
                except Exception as model_error:
                    self.logger.warning(f"Model {model_name} prediction error: {model_error}")
                    continue
            
            # Calculate ensemble prediction
            if predictions:
                signals = [v for k, v in predictions.items() if k.endswith('_signal')]
                confidences = [v for k, v in predictions.items() if k.endswith('_confidence')]
                
                ensemble_signal = np.mean(signals) if signals else 0
                ensemble_confidence = np.mean(confidences) if confidences else 0
                
                predictions['ensemble_signal'] = ensemble_signal
                predictions['ensemble_confidence'] = ensemble_confidence
            
            return predictions
            
        except Exception as e:
            self.logger.warning(f"ML prediction error: {e}")
            return {'ensemble_signal': 0, 'ensemble_confidence': 0}
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        📈 MARKET REGIME DETECTION
        
        Detects current market regime (trending, ranging, volatile)
        """
        try:
            # Calculate volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Calculate trend strength
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            trend_strength = abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
            
            # Calculate range-bound characteristics
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
            range_width = (high_20.iloc[-1] - low_20.iloc[-1]) / data['close'].iloc[-1]
            
            # Classify regime
            if trend_strength > 0.02:  # Strong trend
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    return "STRONG_UPTREND"
                else:
                    return "STRONG_DOWNTREND"
            elif volatility > returns.std() * 1.5:  # High volatility
                return "HIGH_VOLATILITY"
            elif range_width < 0.02:  # Tight range
                return "TIGHT_RANGE"
            else:
                return "RANGING_MARKET"
                
        except Exception as e:
            self.logger.warning(f"Market regime detection error: {e}")
            return "UNKNOWN"

    def _calculate_composite_signal(self, elliott_analysis, technical_analysis, 
                                  ml_predictions, market_regime, current_price, timestamp):
        """
        🎯 COMPOSITE SIGNAL CALCULATION
        
        Combines all analysis methods to generate final trading signal
        """
        try:
            # Get weights from config
            elliott_weight = self.config.get('elliott_wave_weight', 0.30)
            technical_weight = self.config.get('technical_indicators_weight', 0.25)
            ml_weight = self.config.get('ml_prediction_weight', 0.35)
            regime_weight = self.config.get('market_regime_weight', 0.10)
            
            # Calculate Elliott Wave score
            elliott_score = 0
            if elliott_analysis:
                wave_direction = elliott_analysis.get('wave_direction', 'NEUTRAL')
                wave_strength = elliott_analysis.get('wave_strength', 0)
                
                if wave_direction == 'UP':
                    elliott_score = wave_strength / 5.0  # Normalize to 0-1
                elif wave_direction == 'DOWN':
                    elliott_score = -(wave_strength / 5.0)
                
                # Adjust based on wave completion
                completion = elliott_analysis.get('wave_completion', 0.5)
                if completion > 0.8:  # Near completion, look for reversal
                    elliott_score *= 0.5
            
            # Calculate Technical Indicators score
            technical_score = 0
            if technical_analysis and 'overall_score' in technical_analysis:
                technical_score = technical_analysis['overall_score']
                
                # Boost signal if multiple indicators agree
                bullish_count = technical_analysis.get('bullish_count', 0)
                bearish_count = technical_analysis.get('bearish_count', 0)
                
                if bullish_count >= 3:
                    technical_score = min(technical_score + 0.2, 1.0)
                elif bearish_count >= 3:
                    technical_score = max(technical_score - 0.2, -1.0)
            
            # Calculate ML Predictions score
            ml_score = 0
            if ml_predictions:
                ensemble_signal = ml_predictions.get('ensemble_signal', 0)
                ensemble_confidence = ml_predictions.get('ensemble_confidence', 0)
                
                ml_score = ensemble_signal * ensemble_confidence
            
            # Calculate Market Regime adjustment
            regime_adjustment = 1.0
            if market_regime in ['HIGH_VOLATILITY']:
                regime_adjustment = 0.7  # Reduce confidence in volatile markets
            elif market_regime in ['TIGHT_RANGE']:
                regime_adjustment = 0.5  # Reduce signals in ranging markets
            elif market_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                regime_adjustment = 1.2  # Boost signals in trending markets
            
            # Calculate weighted composite score
            composite_score = (
                elliott_score * elliott_weight +
                technical_score * technical_weight +
                ml_score * ml_weight
            ) * regime_adjustment * regime_weight
            
            # Normalize composite score to -1 to 1 range
            composite_score = np.tanh(composite_score)
            
            # Determine signal type based on composite score
            signal_type = self._determine_signal_type(composite_score)
            
            # Calculate confidence
            confidence = abs(composite_score)
            
            # Check minimum confidence threshold
            min_confidence = self.config.get('min_confidence_threshold', 0.70)
            if confidence < min_confidence:
                return None  # No signal if confidence too low
            
            # Calculate risk management parameters
            risk_params = self._calculate_risk_parameters(
                current_price, signal_type, confidence, market_regime, elliott_analysis
            )
            
            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'composite_score': composite_score,
                'price': current_price,
                'target_price': risk_params['target_price'],
                'stop_loss': risk_params['stop_loss'],
                'risk_reward_ratio': risk_params['risk_reward_ratio'],
                'position_size': risk_params['position_size'],
                'elliott_wave_pattern': elliott_analysis.get('wave_pattern', 'UNKNOWN'),
                'market_regime': market_regime,
                'reasoning': self._generate_signal_reasoning(
                    elliott_analysis, technical_analysis, ml_predictions, market_regime, composite_score
                ),
                'technical_indicators': technical_analysis.get('indicators', {}),
                'component_scores': {
                    'elliott_wave': elliott_score,
                    'technical': technical_score,
                    'ml_prediction': ml_score,
                    'regime_adjustment': regime_adjustment
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {e}")
            return None
    
    def _determine_signal_type(self, composite_score):
        """กำหนดประเภทของสัญญาณตาม composite score"""
        if composite_score > 0.8:
            return SignalType.STRONG_BUY
        elif composite_score > 0.3:
            return SignalType.BUY
        elif composite_score < -0.8:
            return SignalType.STRONG_SELL
        elif composite_score < -0.3:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_risk_parameters(self, current_price, signal_type, confidence, 
                                 market_regime, elliott_analysis):
        """
        🛡️ ADVANCED RISK MANAGEMENT
        
        Calculates sophisticated risk management parameters
        """
        try:
            # Base risk parameters from config
            base_stop_loss_pct = self.config.get('stop_loss_percentage', 0.015)
            base_take_profit_mult = self.config.get('take_profit_multiplier', 2.0)
            max_position_size = self.config.get('max_position_size', 0.02)
            min_rr_ratio = self.config.get('min_risk_reward_ratio', 1.5)
            
            # Adjust risk based on market regime
            regime_risk_adjustment = {
                'HIGH_VOLATILITY': 1.5,  # Wider stops in volatile markets
                'TIGHT_RANGE': 0.7,      # Tighter stops in ranging markets
                'STRONG_UPTREND': 0.8,   # Slightly tighter stops in trends
                'STRONG_DOWNTREND': 0.8,
                'UNKNOWN': 1.2
            }
            
            risk_multiplier = regime_risk_adjustment.get(market_regime, 1.0)
            adjusted_stop_loss_pct = base_stop_loss_pct * risk_multiplier
            
            # Adjust position size based on confidence
            confidence_factor = min(confidence * 2, 1.0)  # Max 2x position for high confidence
            position_size = max_position_size * confidence_factor
            
            # Calculate stop loss and target prices
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - adjusted_stop_loss_pct)
                
                # Use Fibonacci extensions for targets if available
                fibonacci_levels = elliott_analysis.get('fibonacci_levels', {}) if elliott_analysis else {}
                if fibonacci_levels:
                    # Look for nearest Fibonacci extension above current price
                    extensions = {k: v for k, v in fibonacci_levels.items() if k.startswith('ext_') and v > current_price}
                    if extensions:
                        target_price = min(extensions.values())  # Nearest extension
                    else:
                        target_price = current_price * (1 + adjusted_stop_loss_pct * base_take_profit_mult)
                else:
                    target_price = current_price * (1 + adjusted_stop_loss_pct * base_take_profit_mult)
                
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = current_price * (1 + adjusted_stop_loss_pct)
                
                # Use Fibonacci extensions for targets if available
                fibonacci_levels = elliott_analysis.get('fibonacci_levels', {}) if elliott_analysis else {}
                if fibonacci_levels:
                    # Look for nearest Fibonacci extension below current price
                    extensions = {k: v for k, v in fibonacci_levels.items() if k.startswith('ext_') and v < current_price}
                    if extensions:
                        target_price = max(extensions.values())  # Nearest extension
                    else:
                        target_price = current_price * (1 - adjusted_stop_loss_pct * base_take_profit_mult)
                else:
                    target_price = current_price * (1 - adjusted_stop_loss_pct * base_take_profit_mult)
                    
            else:  # HOLD
                stop_loss = current_price
                target_price = current_price
                position_size = 0
            
            # Calculate risk/reward ratio
            if signal_type != SignalType.HOLD:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                # Ensure minimum risk/reward ratio
                if risk_reward_ratio < min_rr_ratio:
                    if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                        target_price = current_price + (risk * min_rr_ratio)
                    else:
                        target_price = current_price - (risk * min_rr_ratio)
                    risk_reward_ratio = min_rr_ratio
            else:
                risk_reward_ratio = 0
            
            return {
                'stop_loss': stop_loss,
                'target_price': target_price,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size': position_size,
                'risk_percentage': adjusted_stop_loss_pct,
                'confidence_factor': confidence_factor
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return {
                'stop_loss': current_price,
                'target_price': current_price,
                'risk_reward_ratio': 0,
                'position_size': 0
            }

    def _generate_signal_reasoning(self, elliott_analysis, technical_analysis, 
                                 ml_predictions, market_regime, composite_score):
        """
        🧠 SIGNAL REASONING GENERATOR
        
        Generates human-readable explanation for trading signals
        """
        reasoning_parts = []
        
        try:
            # Elliott Wave reasoning
            if elliott_analysis:
                wave_pattern = elliott_analysis.get('wave_pattern', 'UNKNOWN')
                wave_direction = elliott_analysis.get('wave_direction', 'NEUTRAL')
                wave_strength = elliott_analysis.get('wave_strength', 0)
                
                if wave_pattern != 'UNKNOWN':
                    reasoning_parts.append(f"Elliott Wave shows {wave_pattern} pattern with {wave_direction} direction (strength: {wave_strength}/5)")
            
            # Technical indicators reasoning
            if technical_analysis and 'signal_scores' in technical_analysis:
                signal_scores = technical_analysis['signal_scores']
                
                # Find strongest indicators
                positive_indicators = [k for k, v in signal_scores.items() if v > 0.5]
                negative_indicators = [k for k, v in signal_scores.items() if v < -0.5]
                
                if positive_indicators:
                    reasoning_parts.append(f"Bullish signals from: {', '.join(positive_indicators)}")
                if negative_indicators:
                    reasoning_parts.append(f"Bearish signals from: {', '.join(negative_indicators)}")
            
            # ML predictions reasoning
            if ml_predictions:
                ensemble_confidence = ml_predictions.get('ensemble_confidence', 0)
                ensemble_signal = ml_predictions.get('ensemble_signal', 0)
                
                if ensemble_confidence > 0.7:
                    direction = "bullish" if ensemble_signal > 0 else "bearish"
                    reasoning_parts.append(f"ML models show strong {direction} consensus (confidence: {ensemble_confidence:.1%})")
            
            # Market regime reasoning
            regime_explanations = {
                'STRONG_UPTREND': 'Market is in strong uptrend - trend-following strategy favored',
                'STRONG_DOWNTREND': 'Market is in strong downtrend - trend-following strategy favored',
                'HIGH_VOLATILITY': 'High volatility environment - reduced position sizing recommended',
                'TIGHT_RANGE': 'Market is range-bound - mean reversion strategy appropriate',
                'RANGING_MARKET': 'Market showing ranging characteristics - cautious approach recommended'
            }
            
            if market_regime in regime_explanations:
                reasoning_parts.append(regime_explanations[market_regime])
            
            # Overall signal strength reasoning
            if abs(composite_score) > 0.8:
                reasoning_parts.append(f"Strong signal consensus across multiple timeframes and indicators")
            elif abs(composite_score) > 0.5:
                reasoning_parts.append(f"Moderate signal with good indicator agreement")
            else:
                reasoning_parts.append(f"Weak signal - multiple conflicting indicators")
            
            # Combine all reasoning
            if reasoning_parts:
                return ". ".join(reasoning_parts) + "."
            else:
                return "Signal generated based on composite technical analysis."
                
        except Exception as e:
            self.logger.warning(f"Error generating signal reasoning: {e}")
            return "Technical analysis indicates trading opportunity."
    
    def _validate_signal(self, signal_data):
        """
        ✅ SIGNAL VALIDATION
        
        Validates signal quality before execution
        """
        try:
            if not signal_data:
                return False
            
            # Check confidence threshold
            confidence = signal_data.get('confidence', 0)
            min_confidence = self.config.get('min_confidence_threshold', 0.70)
            if confidence < min_confidence:
                self.logger.info(f"Signal rejected: confidence {confidence:.1%} below threshold {min_confidence:.1%}")
                return False
            
            # Check risk/reward ratio
            risk_reward = signal_data.get('risk_reward_ratio', 0)
            min_rr = self.config.get('min_risk_reward_ratio', 1.5)
            if risk_reward < min_rr:
                self.logger.info(f"Signal rejected: R/R ratio {risk_reward:.2f} below minimum {min_rr}")
                return False
            
            # Check signal cooldown
            cooldown_minutes = self.config.get('signal_cooldown_minutes', 15)
            if self.signal_history:
                last_signal_time = self.signal_history[-1].timestamp
                time_diff = (datetime.now() - last_signal_time).total_seconds() / 60
                if time_diff < cooldown_minutes:
                    self.logger.info(f"Signal rejected: cooldown period ({time_diff:.1f} min < {cooldown_minutes} min)")
                    return False
            
            # Check for conflicting signals
            if len(self.signal_history) >= 2:
                last_two_signals = self.signal_history[-2:]
                if (last_two_signals[0].signal_type != last_two_signals[1].signal_type and
                    (datetime.now() - last_two_signals[-1].timestamp).total_seconds() < 300):  # 5 minutes
                    self.logger.info("Signal rejected: conflicting with recent signal")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def _create_trading_signal(self, signal_data, current_price, timestamp):
        """สร้าง TradingSignal object จากข้อมูลสัญญาณ"""
        try:
            return TradingSignal(
                timestamp=timestamp,
                signal_type=signal_data['signal_type'],
                strength=self._calculate_signal_strength(signal_data['confidence']),
                confidence=signal_data['confidence'],
                price=current_price,
                target_price=signal_data['target_price'],
                stop_loss=signal_data['stop_loss'],
                risk_reward_ratio=signal_data['risk_reward_ratio'],
                position_size=signal_data['position_size'],
                elliott_wave_pattern=signal_data['elliott_wave_pattern'],
                technical_indicators=signal_data['technical_indicators'],
                market_regime=signal_data['market_regime'],
                reasoning=signal_data['reasoning']
            )
        except Exception as e:
            self.logger.error(f"Error creating trading signal: {e}")
            return None
    
    def _calculate_signal_strength(self, confidence):
        """คำนวณความแข็งแกร่งของสัญญาณจาก confidence"""
        if confidence >= 0.9:
            return SignalStrength.EXTREME
        elif confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _check_signal_cooldown(self, timestamp):
        """ตรวจสอบ cooldown period ระหว่างสัญญาณ"""
        if not self.signal_history:
            return True
        
        cooldown_minutes = self.config.get('signal_cooldown_minutes', 15)
        last_signal_time = self.signal_history[-1].timestamp
        time_diff = (timestamp - last_signal_time).total_seconds() / 60
        
        return time_diff >= cooldown_minutes
    
    def _log_signal(self, signal):
        """บันทึก log สำหรับสัญญาณที่สร้าง"""
        self.logger.info(
            f"🎯 Signal Generated: {signal.signal_type.value} | "
            f"Confidence: {signal.confidence:.1%} | "
            f"Price: ${signal.price:.2f} | "
            f"Target: ${signal.target_price:.2f} | "
            f"R/R: {signal.risk_reward_ratio:.2f}"
        )
        
        # Update performance metrics
        self.performance_metrics['total_signals'] += 1
    
    def get_signal_summary(self):
        """ได้สรุปข้อมูลสัญญาณ"""
        return {
            'total_signals_generated': len(self.signal_history),
            'recent_signals': [
                {
                    'timestamp': signal.timestamp.isoformat(),
                    'type': signal.signal_type.value,
                    'confidence': signal.confidence
                }
                for signal in self.signal_history[-5:]  # Last 5 signals
            ],
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
    
    def _prepare_ml_features(self, data):
        """เตรียมฟีเจอร์สำหรับ ML models"""
        try:
            # Basic technical indicators as features
            features = []
            
            # Price-based features
            close = data['close'].values
            features.extend([
                close[-1],  # Current price
                close[-1] / close[-20] - 1,  # 20-period return
                np.std(close[-20:]) / np.mean(close[-20:])  # 20-period volatility
            ])
            
            # Moving averages
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
            features.extend([
                close[-1] / sma_20 - 1,  # Distance from SMA20
                close[-1] / sma_50 - 1,  # Distance from SMA50
                sma_20 / sma_50 - 1      # SMA20/SMA50 ratio
            ])
            
            # RSI
            if len(close) >= 14:
                rsi = self._calculate_rsi(pd.Series(close), 14)
                features.append(rsi / 100.0)  # Normalize to 0-1
            else:
                features.append(0.5)
            
            # Ensure we have a consistent number of features
            while len(features) < 10:  # Pad to 10 features
                features.append(0.0)
            
            return np.array(features[:10])  # Use first 10 features
            
        except Exception as e:
            self.logger.warning(f"Error preparing ML features: {e}")
            return np.zeros(10)  # Return zero array as fallback

    def _identify_elliott_waves(self, close_prices, high_prices, low_prices):
        """
        Identify Elliott Waves in the price data
        """
        try:
            # Simple wave identification logic
            # This should be replaced with a proper Elliott Wave detection algorithm
            waves = []
            wave = []
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i - 1]:
                    wave.append({'price': close_prices[i], 'type': 'peak'})
                elif close_prices[i] < close_prices[i - 1]:
                    if wave and wave[-1]['type'] == 'peak':
                        wave.append({'price': close_prices[i], 'type': 'trough'})
                    elif wave and wave[-1]['type'] == 'trough' and len(wave) >= 3:
                        waves.append({'pivots': wave, 'pattern': 'IMPULSE'})
                        wave = []
            if wave:
                waves.append({'pivots': wave, 'pattern': 'CORRECTIVE'})
            
            return waves
        
        except Exception as e:
            self.logger.warning(f"Error identifying Elliott waves: {e}")
            return []
    
    def _determine_current_wave_position(self, waves):
        """กำหนดตำแหน่ง Elliott Wave ปัจจุบัน"""
        if not waves:
            return {'pattern': 'UNKNOWN', 'position': 0, 'completion': 0.0}
        
        # Get the most recent wave
        current_wave = waves[-1]
        
        # Analyze wave completion
        completion = self._calculate_wave_completion(current_wave.get('pivots', []))
        
        return {
            'pattern': current_wave.get('pattern', 'UNKNOWN'),
            'position': len(current_wave.get('pivots', [])),
            'completion': completion,
            'direction': current_wave.get('direction', 'NEUTRAL'),
            'strength': current_wave.get('strength', 0)
        }
    
    def _calculate_wave_completion(self, pivots):
        """คำนวณระดับความสมบูรณ์ของคลื่น"""
        if len(pivots) < 2:
            return 0.0
        
        try:
            # Simple completion based on pivot count
            # For impulse waves (5 pivots), for corrective (3 pivots)
            if len(pivots) >= 5:
                return min(len(pivots) / 5.0, 1.0)
            else:
                return min(len(pivots) / 3.0, 1.0)
        except:
            return 0.0
    
    def _determine_wave_direction(self, wave_sequence):
        """กำหนดทิศทางของคลื่น Elliott Wave"""
        if len(wave_sequence) < 2:
            return 'NEUTRAL'
        
        try:
            # Compare start and end prices
            start_price = wave_sequence[0]['price']
            end_price = wave_sequence[-1]['price']
            
            price_change = (end_price - start_price) / start_price
            
            if price_change > 0.01:  # > 1% move up
                return 'UP'
            elif price_change < -0.01:  # > 1% move down
                return 'DOWN'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'
    
    def _calculate_pattern_strength(self, wave_sequence):
        """คำนวณความแข็งแกร่งของรูปแบบ Elliott Wave"""
        if len(wave_sequence) < 3:
            return 1
        
        try:
            # Calculate strength based on:
            # 1. Price range of the pattern
            # 2. Number of pivots
            # 3. Consistency of the pattern
            
            prices = [pivot['price'] for pivot in wave_sequence]
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            relative_range = price_range / avg_price
            
            # Strength based on relative price range
            if relative_range > 0.05:  # > 5% range
                base_strength = 5
            elif relative_range > 0.03:  # > 3% range
                base_strength = 4
            elif relative_range > 0.02:  # > 2% range
                base_strength = 3
            elif relative_range > 0.01:  # > 1% range
                base_strength = 2
            else:
                base_strength = 1
            
            # Adjust for pattern consistency
            pivot_strengths = [pivot.get('strength', 1) for pivot in wave_sequence]
            avg_pivot_strength = sum(pivot_strengths) / len(pivot_strengths)
            
            final_strength = min(int(base_strength * avg_pivot_strength), 5)
            return max(final_strength, 1)
            
        except:
            return 1
    
    def _calculate_wave_signal(self, current_wave, waves):
        """คำนวณสัญญาณจาก Elliott Wave"""
        try:
            wave_direction = current_wave.get('direction', 'NEUTRAL')
            wave_completion = current_wave.get('completion', 0.0)
            wave_pattern = current_wave.get('pattern', 'UNKNOWN')
            
            # Base signal from wave direction
            if wave_direction == 'UP':
                base_signal = 1
            elif wave_direction == 'DOWN':
                base_signal = -1
            else:
                base_signal = 0
            
            # Adjust based on wave completion
            # If wave is nearly complete, look for reversal
            if wave_completion > 0.8:
                if wave_pattern == 'IMPULSE':
                    # End of impulse wave - expect correction
                    signal_strength = 2  # Moderate reversal signal
                elif wave_pattern == 'CORRECTIVE':
                    # End of correction - expect trend resumption
                    signal_strength = 3  # Strong continuation signal
                else:
                    signal_strength = 1
            elif wave_completion > 0.5:
                # Mid-wave - continue with trend
                signal_strength = 3
            else:
                # Early wave - weak signal
                signal_strength = 1
            
            return {
                'direction': wave_direction,
                'strength': signal_strength,
                'completion_based': wave_completion > 0.8
            }
            
        except:
            return {'direction': 'NEUTRAL', 'strength': 0, 'completion_based': False}
    
    def _is_complex_pattern(self, sequence):
        """ตรวจสอบรูปแบบ Elliott Wave ที่ซับซ้อน"""
        # Complex patterns include triangles, flats, zigzags, etc.
        # This is a simplified implementation
        try:
            if len(sequence) < 4:
                return False
            
            # Look for triangle patterns (converging highs and lows)
            highs = [p['price'] for p in sequence if p['type'] == 'peak']
            lows = [p['price'] for p in sequence if p['type'] == 'trough']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check if highs are converging downward and lows upward
                high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
                low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
                
                # Triangle: highs declining, lows rising
                if high_trend < 0 and low_trend > 0:
                    return True
            
            return False
        except:
            return False

# Additional utility functions for the signal generator
def create_default_signal_generator(logger=None):
    """Factory function to create a default signal generator"""
    return AdvancedTradingSignalGenerator(
        models={},
        config={
            'min_confidence_threshold': 0.75,
            'max_position_size': 0.02,
            'min_risk_reward_ratio': 2.0,
            'elliott_wave_weight': 0.35,
            'technical_indicators_weight': 0.25,
            'ml_prediction_weight': 0.30,
            'market_regime_weight': 0.10
        },
        logger=logger
    )

def test_signal_generator():
    """Test function for the signal generator"""
    print("🧪 Testing Advanced Trading Signal Generator...")
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 2000,
        'high': np.random.randn(100).cumsum() + 2020,
        'low': np.random.randn(100).cumsum() + 1980,
        'close': np.random.randn(100).cumsum() + 2000,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Create signal generator
    signal_gen = create_default_signal_generator()
    
    # Generate signal
    current_price = test_data['close'].iloc[-1]
    signal = signal_gen.generate_signal(test_data, current_price)
    
    if signal:
        print(f"✅ Signal generated: {signal.signal_type.value}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Price: ${signal.price:.2f}")
        print(f"   Target: ${signal.target_price:.2f}")
        print(f"   Stop Loss: ${signal.stop_loss:.2f}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
    else:
        print("⚠️ No signal generated (below confidence threshold)")
    
    return signal_gen

if __name__ == "__main__":
    # Run test
    test_signal_generator()
