#!/usr/bin/env python3
"""
🌊 ADVANCED ELLIOTT WAVE ANALYZER
ระบบวิเคราะห์ Elliott Wave ขั้นสูงพร้อม Multi-Timeframe และ Wave Counting

Advanced Features:
- Multi-Timeframe Wave Detection (M1, M5, M15, H1, H4, D1)
- Impulse/Corrective Wave Classification
- Fibonacci-based Wave Measurement
- Cross-Timeframe Wave Correlation
- Wave Position Identification (Wave 1-5, A-B-C)
- Advanced Elliott Wave Trading Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Advanced Logging Integration
try:
    from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False


class WaveType(Enum):
    """ประเภทของ Elliott Wave"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    UNKNOWN = "unknown"


class WavePosition(Enum):
    """ตำแหน่งของ Wave ใน Elliott Wave Cycle"""
    WAVE_1 = "wave_1"
    WAVE_2 = "wave_2"
    WAVE_3 = "wave_3"
    WAVE_4 = "wave_4"
    WAVE_5 = "wave_5"
    WAVE_A = "wave_a"
    WAVE_B = "wave_b"
    WAVE_C = "wave_c"
    UNKNOWN = "unknown"


class WaveDirection(Enum):
    """ทิศทางของ Wave"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class ElliottWave:
    """ข้อมูล Elliott Wave แต่ละคลื่น"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    wave_type: WaveType
    wave_position: WavePosition
    direction: WaveDirection
    timeframe: str
    fibonacci_ratio: float
    confidence: float
    volume_confirmation: bool
    
    @property
    def price_change(self) -> float:
        return self.end_price - self.start_price
    
    @property
    def price_change_pct(self) -> float:
        return (self.end_price - self.start_price) / self.start_price if self.start_price != 0 else 0


@dataclass
class MultiTimeframeWaveAnalysis:
    """การวิเคราะห์ Wave หลายกรอบเวลา"""
    primary_wave: ElliottWave
    supporting_waves: Dict[str, ElliottWave]  # timeframe -> wave
    wave_alignment_score: float
    fibonacci_confluence: List[float]
    trend_strength: float
    recommended_action: str
    confidence_score: float


class AdvancedElliottWaveAnalyzer:
    """ตัววิเคราะห์ Elliott Wave ขั้นสูง"""
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        
        # Initialize Advanced Logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_unified_logger()
            self.logger.info("🚀 AdvancedElliottWaveAnalyzer initialized")
        else:
            self.logger = logger or get_unified_logger()
        
        # Configuration
        self.timeframes = self.config.get('timeframes', ['1min', '5min', '15min', '1H', '4H', '1D'])
        self.primary_timeframe = self.config.get('primary_timeframe', '1min')
        self.wave_detection_sensitivity = self.config.get('wave_detection_sensitivity', 0.02)  # 2%
        self.fibonacci_tolerance = self.config.get('fibonacci_tolerance', 0.05)  # 5%
        
        # Fibonacci ratios for Elliott Wave analysis
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.0, 1.272, 1.414, 1.618, 2.0, 2.618],
            'projection': [0.618, 1.0, 1.382, 1.618]
        }
        
        # Wave patterns and rules
        self.elliott_wave_rules = self._initialize_elliott_wave_rules()
        
        # Cache for multi-timeframe data
        self.timeframe_cache = {}
        
    def _initialize_elliott_wave_rules(self) -> Dict[str, Any]:
        """เริ่มต้นกฎของ Elliott Wave Theory"""
        return {
            'impulse_rules': {
                'wave_2_cannot_retrace_below_wave_1_start': True,
                'wave_3_cannot_be_shortest': True,
                'wave_4_cannot_overlap_wave_1': True,
                'wave_5_fibonacci_relationships': [0.618, 1.0, 1.618]
            },
            'corrective_rules': {
                'abc_pattern': True,
                'wave_b_retracement_levels': [0.382, 0.5, 0.618],
                'wave_c_projection_levels': [1.0, 1.272, 1.618]
            },
            'fibonacci_confluence': {
                'minimum_confluences': 2,
                'confluence_tolerance': 0.01
            }
        }
    
    def analyze_multi_timeframe_waves(self, data: pd.DataFrame) -> MultiTimeframeWaveAnalysis:
        """วิเคราะห์ Elliott Wave หลายกรอบเวลา"""
        try:
            self.logger.info("🌊 Starting multi-timeframe Elliott Wave analysis...")
            
            # Prepare multi-timeframe data
            timeframe_data = self._prepare_multi_timeframe_data(data)
            
            # Detect waves in each timeframe
            timeframe_waves = {}
            for tf, tf_data in timeframe_data.items():
                waves = self._detect_elliott_waves(tf_data, tf)
                if waves:
                    timeframe_waves[tf] = waves[-1]  # Get most recent wave
                    self.logger.info(f"📊 Detected wave in {tf}: {waves[-1].wave_position.value}")
            
            if not timeframe_waves:
                self.logger.warning("⚠️ No waves detected in any timeframe")
                return self._create_empty_analysis()
            
            # Identify primary wave (from primary timeframe or strongest signal)
            primary_wave = self._identify_primary_wave(timeframe_waves)
            
            # Calculate wave alignment score
            alignment_score = self._calculate_wave_alignment(timeframe_waves, primary_wave)
            
            # Find Fibonacci confluences across timeframes
            fibonacci_confluences = self._find_fibonacci_confluences(timeframe_waves)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(timeframe_waves)
            
            # Generate trading recommendation
            recommendation = self._generate_wave_based_recommendation(
                primary_wave, timeframe_waves, alignment_score, fibonacci_confluences
            )
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                alignment_score, len(fibonacci_confluences), trend_strength
            )
            
            analysis = MultiTimeframeWaveAnalysis(
                primary_wave=primary_wave,
                supporting_waves={k: v for k, v in timeframe_waves.items() if k != primary_wave.timeframe},
                wave_alignment_score=alignment_score,
                fibonacci_confluence=fibonacci_confluences,
                trend_strength=trend_strength,
                recommended_action=recommendation,
                confidence_score=confidence
            )
            
            self.logger.info(f"✅ Multi-timeframe analysis complete - Confidence: {confidence:.2f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe wave analysis failed: {str(e)}")
            return self._create_empty_analysis()
    
    def _prepare_multi_timeframe_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """เตรียมข้อมูลหลายกรอบเวลา"""
        timeframe_data = {}
        
        # Ensure timestamp column exists
        if 'timestamp' not in data.columns:
            if 'Date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['Date'])
            elif data.index.name == 'timestamp' or hasattr(data.index, 'to_datetime'):
                data = data.reset_index()
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            else:
                # Create synthetic timestamp
                data['timestamp'] = pd.date_range(
                    start='2024-01-01', periods=len(data), freq='1min'
                )
        
        # Resample to different timeframes
        base_data = data.set_index('timestamp')
        
        for tf in self.timeframes:
            try:
                # Convert timeframe notation
                resample_rule = self._convert_timeframe_to_pandas(tf)
                
                # Resample data
                resampled = base_data.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum' if 'volume' in base_data.columns else 'mean'
                }).dropna()
                
                # Add basic indicators for wave detection
                resampled = self._add_wave_detection_indicators(resampled)
                
                timeframe_data[tf] = resampled.reset_index()
                
                self.logger.info(f"📈 Prepared {tf} data: {len(resampled)} bars")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to prepare {tf} data: {str(e)}")
        
        return timeframe_data
    
    def _convert_timeframe_to_pandas(self, timeframe: str) -> str:
        """แปลง timeframe notation เป็น pandas resample rule"""
        tf_mapping = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D'
        }
        return tf_mapping.get(timeframe, timeframe)
    
    def _add_wave_detection_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """เพิ่ม indicators สำหรับการตรวจจับ wave"""
        # Pivot points for wave detection
        data['pivot_high'] = self._find_pivot_highs(data['high'])
        data['pivot_low'] = self._find_pivot_lows(data['low'])
        
        # Trend indicators
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
        # Momentum indicators
        data['rsi'] = self._calculate_rsi(data['close'])
        data['macd'] = self._calculate_macd(data['close'])
        
        # Volume confirmation
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        return data
    
    def _detect_elliott_waves(self, data: pd.DataFrame, timeframe: str) -> List[ElliottWave]:
        """ตรวจจับ Elliott Waves ในกรอบเวลาหนึ่ง"""
        waves = []
        
        try:
            # Find swing points (pivots)
            swing_highs = data[data['pivot_high'] == 1].copy()
            swing_lows = data[data['pivot_low'] == 1].copy()
            
            # Combine and sort swing points
            swing_points = []
            
            for _, high in swing_highs.iterrows():
                swing_points.append({
                    'timestamp': high['timestamp'],
                    'price': high['high'],
                    'type': 'high'
                })
            
            for _, low in swing_lows.iterrows():
                swing_points.append({
                    'timestamp': low['timestamp'],
                    'price': low['low'],
                    'type': 'low'
                })
            
            # Sort by timestamp
            swing_points = sorted(swing_points, key=lambda x: x['timestamp'])
            
            if len(swing_points) < 6:  # Need at least 6 points for 5-wave pattern
                return waves
            
            # Identify wave patterns
            waves = self._identify_wave_patterns(swing_points, timeframe, data)
            
            self.logger.info(f"🌊 Detected {len(waves)} waves in {timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ Wave detection failed for {timeframe}: {str(e)}")
        
        return waves
    
    def _identify_wave_patterns(self, swing_points: List[Dict], timeframe: str, data: pd.DataFrame) -> List[ElliottWave]:
        """ระบุรูปแบบ wave จาก swing points"""
        waves = []
        
        # Look for 5-wave impulse patterns
        for i in range(len(swing_points) - 5):
            pattern = swing_points[i:i+6]
            
            # Check if this could be a 5-wave pattern
            if self._is_valid_impulse_pattern(pattern):
                # Create waves for this pattern
                for j in range(5):
                    wave = self._create_wave_from_pattern(
                        pattern[j], pattern[j+1], j+1, WaveType.IMPULSE, timeframe, data
                    )
                    if wave:
                        waves.append(wave)
        
        # Look for 3-wave corrective patterns
        for i in range(len(swing_points) - 3):
            pattern = swing_points[i:i+4]
            
            if self._is_valid_corrective_pattern(pattern):
                # Create ABC correction waves
                wave_positions = [WavePosition.WAVE_A, WavePosition.WAVE_B, WavePosition.WAVE_C]
                for j in range(3):
                    wave = self._create_wave_from_pattern(
                        pattern[j], pattern[j+1], wave_positions[j], WaveType.CORRECTIVE, timeframe, data
                    )
                    if wave:
                        waves.append(wave)
        
        return waves
    
    def _is_valid_impulse_pattern(self, pattern: List[Dict]) -> bool:
        """ตรวจสอบว่าเป็น impulse pattern ที่ถูกต้องหรือไม่"""
        if len(pattern) != 6:
            return False
        
        try:
            # Extract prices
            prices = [p['price'] for p in pattern]
            
            # Check for basic impulse structure
            # Pattern should be: Low-High-Low-High-Low-High (uptrend)
            # or High-Low-High-Low-High-Low (downtrend)
            
            # Determine trend direction
            start_price = prices[0]
            end_price = prices[-1]
            is_uptrend = end_price > start_price
            
            if is_uptrend:
                # Should start with low, then alternate high-low
                expected_types = ['low', 'high', 'low', 'high', 'low', 'high']
            else:
                # Should start with high, then alternate low-high
                expected_types = ['high', 'low', 'high', 'low', 'high', 'low']
            
            # Check pattern structure
            for i, point in enumerate(pattern):
                if i < len(expected_types) and point['type'] != expected_types[i]:
                    return False
            
            # Apply Elliott Wave rules
            if is_uptrend:
                # Wave 2 should not go below wave 1 start
                if prices[2] <= prices[0]:
                    return False
                
                # Wave 4 should not overlap wave 1
                if prices[4] <= prices[1]:
                    return False
            else:
                # For downtrend, reverse the logic
                if prices[2] >= prices[0]:
                    return False
                
                if prices[4] >= prices[1]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_corrective_pattern(self, pattern: List[Dict]) -> bool:
        """ตรวจสอบว่าเป็น corrective pattern ที่ถูกต้องหรือไม่"""
        if len(pattern) != 4:
            return False
        
        try:
            # Basic ABC correction structure
            prices = [p['price'] for p in pattern]
            
            # Check for 3-wave structure
            # Should be alternating highs and lows
            types = [p['type'] for p in pattern]
            
            # Valid patterns: high-low-high-low or low-high-low-high
            valid_patterns = [
                ['high', 'low', 'high', 'low'],
                ['low', 'high', 'low', 'high']
            ]
            
            return types in valid_patterns
            
        except Exception:
            return False
    
    def _create_wave_from_pattern(self, start_point: Dict, end_point: Dict, 
                                wave_num: int, wave_type: WaveType, 
                                timeframe: str, data: pd.DataFrame) -> Optional[ElliottWave]:
        """สร้าง ElliottWave object จาก pattern"""
        try:
            # Determine wave position
            if wave_type == WaveType.IMPULSE:
                wave_positions = [
                    WavePosition.WAVE_1, WavePosition.WAVE_2, WavePosition.WAVE_3,
                    WavePosition.WAVE_4, WavePosition.WAVE_5
                ]
                wave_position = wave_positions[wave_num - 1] if wave_num <= 5 else WavePosition.UNKNOWN
            else:
                wave_position = wave_num  # Already passed as WavePosition enum
            
            # Determine direction
            price_change = end_point['price'] - start_point['price']
            if price_change > self.wave_detection_sensitivity * start_point['price']:
                direction = WaveDirection.UP
            elif price_change < -self.wave_detection_sensitivity * start_point['price']:
                direction = WaveDirection.DOWN
            else:
                direction = WaveDirection.SIDEWAYS
            
            # Calculate Fibonacci ratio (simplified)
            fibonacci_ratio = self._calculate_fibonacci_ratio(start_point, end_point)
            
            # Calculate confidence based on various factors
            confidence = self._calculate_wave_confidence(start_point, end_point, data)
            
            # Check volume confirmation
            volume_confirmation = self._check_volume_confirmation(
                start_point['timestamp'], end_point['timestamp'], data
            )
            
            wave = ElliottWave(
                start_time=start_point['timestamp'],
                end_time=end_point['timestamp'],
                start_price=start_point['price'],
                end_price=end_point['price'],
                wave_type=wave_type,
                wave_position=wave_position,
                direction=direction,
                timeframe=timeframe,
                fibonacci_ratio=fibonacci_ratio,
                confidence=confidence,
                volume_confirmation=volume_confirmation
            )
            
            return wave
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create wave: {str(e)}")
            return None
    
    def _calculate_fibonacci_ratio(self, start_point: Dict, end_point: Dict) -> float:
        """คำนวณ Fibonacci ratio ของ wave"""
        price_move = abs(end_point['price'] - start_point['price'])
        
        # Compare with common Fibonacci ratios
        base_price = start_point['price']
        
        for ratio in self.fibonacci_ratios['retracement'] + self.fibonacci_ratios['extension']:
            expected_move = base_price * ratio
            if abs(price_move - expected_move) / expected_move < self.fibonacci_tolerance:
                return ratio
        
        # Return actual ratio if no match
        return price_move / base_price if base_price != 0 else 0.0
    
    def _calculate_wave_confidence(self, start_point: Dict, end_point: Dict, data: pd.DataFrame) -> float:
        """คำนวณความน่าเชื่อถือของ wave"""
        confidence = 0.5  # Base confidence
        
        try:
            # Price movement significance
            price_change_pct = abs(end_point['price'] - start_point['price']) / start_point['price']
            if price_change_pct > 0.02:  # > 2%
                confidence += 0.2
            
            # Time duration
            time_diff = (end_point['timestamp'] - start_point['timestamp']).total_seconds()
            if time_diff > 3600:  # > 1 hour
                confidence += 0.1
            
            # Fibonacci alignment
            fib_ratio = self._calculate_fibonacci_ratio(start_point, end_point)
            if fib_ratio in self.fibonacci_ratios['retracement'] + self.fibonacci_ratios['extension']:
                confidence += 0.2
            
        except Exception:
            pass
        
        return min(confidence, 1.0)
    
    def _check_volume_confirmation(self, start_time: datetime, end_time: datetime, data: pd.DataFrame) -> bool:
        """ตรวจสอบการยืนยันด้วย volume"""
        try:
            if 'volume' not in data.columns or 'volume_ratio' not in data.columns:
                return False
            
            # Get data for this wave period
            wave_data = data[
                (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
            ]
            
            if len(wave_data) == 0:
                return False
            
            # Check if average volume during wave is above normal
            avg_volume_ratio = wave_data['volume_ratio'].mean()
            return avg_volume_ratio > 1.2  # 20% above average
            
        except Exception:
            return False
    
    def _find_pivot_highs(self, series: pd.Series, window: int = 5) -> pd.Series:
        """หา pivot highs"""
        pivots = pd.Series(0, index=series.index)
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
                pivots.iloc[i] = 1
        
        return pivots
    
    def _find_pivot_lows(self, series: pd.Series, window: int = 5) -> pd.Series:
        """หา pivot lows"""
        pivots = pd.Series(0, index=series.index)
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1)):
                pivots.iloc[i] = 1
        
        return pivots
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """คำนวณ RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """คำนวณ MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26
    
    def _identify_primary_wave(self, timeframe_waves: Dict[str, ElliottWave]) -> ElliottWave:
        """ระบุ primary wave จาก timeframe ต่างๆ"""
        # Priority: primary timeframe > highest confidence > longest timeframe
        
        if self.primary_timeframe in timeframe_waves:
            return timeframe_waves[self.primary_timeframe]
        
        # Choose by confidence
        best_wave = max(timeframe_waves.values(), key=lambda w: w.confidence)
        return best_wave
    
    def _calculate_wave_alignment(self, timeframe_waves: Dict[str, ElliottWave], 
                                primary_wave: ElliottWave) -> float:
        """คำนวณ wave alignment score"""
        if len(timeframe_waves) <= 1:
            return 1.0
        
        alignment_score = 0.0
        total_comparisons = 0
        
        for tf, wave in timeframe_waves.items():
            if wave == primary_wave:
                continue
            
            # Check direction alignment
            if wave.direction == primary_wave.direction:
                alignment_score += 0.5
            
            # Check wave type alignment
            if wave.wave_type == primary_wave.wave_type:
                alignment_score += 0.3
            
            # Check Fibonacci alignment
            if abs(wave.fibonacci_ratio - primary_wave.fibonacci_ratio) < self.fibonacci_tolerance:
                alignment_score += 0.2
            
            total_comparisons += 1
        
        return alignment_score / total_comparisons if total_comparisons > 0 else 0.0
    
    def _find_fibonacci_confluences(self, timeframe_waves: Dict[str, ElliottWave]) -> List[float]:
        """หา Fibonacci confluences ระหว่างกรอบเวลา"""
        confluences = []
        
        # Collect all Fibonacci levels from all waves
        fib_levels = []
        for wave in timeframe_waves.values():
            if wave.fibonacci_ratio in self.fibonacci_ratios['retracement'] + self.fibonacci_ratios['extension']:
                fib_levels.append(wave.fibonacci_ratio)
        
        # Find levels that appear multiple times
        for ratio in set(fib_levels):
            count = fib_levels.count(ratio)
            if count >= 2:
                confluences.append(ratio)
        
        return confluences
    
    def _calculate_trend_strength(self, timeframe_waves: Dict[str, ElliottWave]) -> float:
        """คำนวณความแข็งแกร่งของเทรนด์"""
        if not timeframe_waves:
            return 0.0
        
        # Count waves in same direction
        directions = [wave.direction for wave in timeframe_waves.values()]
        up_count = directions.count(WaveDirection.UP)
        down_count = directions.count(WaveDirection.DOWN)
        
        total_waves = len(directions)
        dominant_direction_count = max(up_count, down_count)
        
        return dominant_direction_count / total_waves if total_waves > 0 else 0.0
    
    def _generate_wave_based_recommendation(self, primary_wave: ElliottWave, 
                                         timeframe_waves: Dict[str, ElliottWave],
                                         alignment_score: float,
                                         fibonacci_confluences: List[float]) -> str:
        """สร้างคำแนะนำการเทรดจาก wave analysis"""
        
        # Strong alignment and confluence = strong signal
        if alignment_score > 0.7 and len(fibonacci_confluences) >= 2:
            if primary_wave.direction == WaveDirection.UP:
                return "STRONG_BUY"
            elif primary_wave.direction == WaveDirection.DOWN:
                return "STRONG_SELL"
        
        # Moderate signals
        if alignment_score > 0.5:
            if primary_wave.direction == WaveDirection.UP:
                return "BUY"
            elif primary_wave.direction == WaveDirection.DOWN:
                return "SELL"
        
        return "HOLD"
    
    def _calculate_overall_confidence(self, alignment_score: float, 
                                   confluence_count: int, trend_strength: float) -> float:
        """คำนวณความเชื่อมั่นโดยรวม"""
        confidence = 0.0
        
        # Alignment contribution (40%)
        confidence += alignment_score * 0.4
        
        # Confluence contribution (30%)
        confluence_score = min(confluence_count / 3, 1.0)  # Max at 3 confluences
        confidence += confluence_score * 0.3
        
        # Trend strength contribution (30%)
        confidence += trend_strength * 0.3
        
        return min(confidence, 1.0)
    
    def _create_empty_analysis(self) -> MultiTimeframeWaveAnalysis:
        """สร้าง empty analysis เมื่อไม่มีข้อมูล"""
        empty_wave = ElliottWave(
            start_time=datetime.now(),
            end_time=datetime.now(),
            start_price=0.0,
            end_price=0.0,
            wave_type=WaveType.UNKNOWN,
            wave_position=WavePosition.UNKNOWN,
            direction=WaveDirection.SIDEWAYS,
            timeframe="unknown",
            fibonacci_ratio=0.0,
            confidence=0.0,
            volume_confirmation=False
        )
        
        return MultiTimeframeWaveAnalysis(
            primary_wave=empty_wave,
            supporting_waves={},
            wave_alignment_score=0.0,
            fibonacci_confluence=[],
            trend_strength=0.0,
            recommended_action="HOLD",
            confidence_score=0.0
        )
    
    def generate_elliott_wave_features(self, analysis: MultiTimeframeWaveAnalysis) -> Dict[str, float]:
        """สร้างฟีเจอร์สำหรับ ML models จาก Elliott Wave analysis"""
        features = {}
        
        # Primary wave features
        pw = analysis.primary_wave
        features['primary_wave_direction'] = 1.0 if pw.direction == WaveDirection.UP else -1.0 if pw.direction == WaveDirection.DOWN else 0.0
        features['primary_wave_confidence'] = pw.confidence
        features['primary_wave_fibonacci_ratio'] = pw.fibonacci_ratio
        features['primary_wave_price_change_pct'] = pw.price_change_pct
        features['primary_wave_volume_confirmation'] = 1.0 if pw.volume_confirmation else 0.0
        
        # Wave type encoding
        features['primary_wave_is_impulse'] = 1.0 if pw.wave_type == WaveType.IMPULSE else 0.0
        features['primary_wave_is_corrective'] = 1.0 if pw.wave_type == WaveType.CORRECTIVE else 0.0
        
        # Wave position encoding (one-hot style)
        wave_positions = [WavePosition.WAVE_1, WavePosition.WAVE_2, WavePosition.WAVE_3,
                         WavePosition.WAVE_4, WavePosition.WAVE_5, WavePosition.WAVE_A,
                         WavePosition.WAVE_B, WavePosition.WAVE_C]
        
        for pos in wave_positions:
            features[f'primary_wave_is_{pos.value}'] = 1.0 if pw.wave_position == pos else 0.0
        
        # Multi-timeframe features
        features['wave_alignment_score'] = analysis.wave_alignment_score
        features['fibonacci_confluence_count'] = float(len(analysis.fibonacci_confluence))
        features['trend_strength'] = analysis.trend_strength
        features['overall_confidence'] = analysis.confidence_score
        
        # Supporting waves count by direction
        supporting_up = sum(1 for w in analysis.supporting_waves.values() if w.direction == WaveDirection.UP)
        supporting_down = sum(1 for w in analysis.supporting_waves.values() if w.direction == WaveDirection.DOWN)
        
        features['supporting_waves_up_count'] = float(supporting_up)
        features['supporting_waves_down_count'] = float(supporting_down)
        features['supporting_waves_net_direction'] = float(supporting_up - supporting_down)
        
        # Action encoding
        action_map = {"STRONG_BUY": 1.0, "BUY": 0.5, "HOLD": 0.0, "SELL": -0.5, "STRONG_SELL": -1.0}
        features['recommended_action_numeric'] = action_map.get(analysis.recommended_action, 0.0)
        
        return features


# Example usage function
def create_advanced_elliott_wave_analyzer(config: Dict = None) -> AdvancedElliottWaveAnalyzer:
    """สร้าง AdvancedElliottWaveAnalyzer instance"""
    default_config = {
        'timeframes': ['1min', '5min', '15min', '1H', '4H'],
        'primary_timeframe': '1min',
        'wave_detection_sensitivity': 0.02,
        'fibonacci_tolerance': 0.05
    }
    
    if config:
        default_config.update(config)
    
    return AdvancedElliottWaveAnalyzer(config=default_config)


if __name__ == "__main__":
    # Test the analyzer
    print("🧪 Testing Advanced Elliott Wave Analyzer")
    analyzer = create_advanced_elliott_wave_analyzer()
    print("✅ Analyzer created successfully")
