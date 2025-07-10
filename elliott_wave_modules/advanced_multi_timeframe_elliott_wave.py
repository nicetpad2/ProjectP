#!/usr/bin/env python3
"""
🌊 ADVANCED MULTI-TIMEFRAME ELLIOTT WAVE ANALYZER
ระบบวิเคราะห์ Elliott Wave แบบหลายไทม์เฟรมขั้นสูง

Advanced Features:
- Multi-Timeframe Elliott Wave Detection (M1, M5, M15, M30, H1, H4, D1)
- Fractal Elliott Wave Analysis
- Wave Confluence Analysis
- Advanced Fibonacci Projections
- Elliott Wave Cycle Analysis
- Wave Personality Recognition
- Market Structure Analysis
"""

import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.project_paths import get_project_paths
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


class AdvancedMultiTimeframeElliottWaveAnalyzer:
    """ระบบวิเคราะห์ Elliott Wave แบบหลายไทม์เฟรมขั้นสูง"""
    
    def __init__(self):
        """Initialize Advanced Multi-Timeframe Elliott Wave Analyzer"""
        self.paths = get_project_paths()
        self.logger = get_unified_logger()
        
        # Elliott Wave Configuration
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective': ['A', 'B', 'C'],
            'complex_corrective': ['W', 'X', 'Y', 'Z']
        }
        
        # Fibonacci Ratios for Elliott Wave
        self.fibonacci_ratios = {
            'retracements': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extensions': [1.0, 1.272, 1.382, 1.618, 2.618],
            'projections': [0.618, 1.0, 1.272, 1.618, 2.618]
        }
        
        # Wave Personality Rules
        self.wave_rules = {
            'wave_1': {'length': 'medium', 'volume': 'increasing', 'momentum': 'building'},
            'wave_2': {'retracement': [0.382, 0.618], 'volume': 'decreasing', 'time': 'long'},
            'wave_3': {'length': 'longest', 'volume': 'highest', 'momentum': 'strongest'},
            'wave_4': {'retracement': [0.236, 0.382], 'volume': 'low', 'time': 'long'},
            'wave_5': {'length': 'similar_to_1', 'volume': 'divergence', 'momentum': 'weak'}
        }
    
    def analyze_multi_timeframe_elliott_waves(self, data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ Elliott Wave แบบหลายไทม์เฟรม"""
        try:
            results = {
                'timeframe_analysis': {},
                'confluence_analysis': {},
                'primary_wave_count': None,
                'trend_direction': None,
                'wave_strength': 0,
                'fibonacci_levels': {},
                'trading_signals': []
            }
            
            # สร้างข้อมูลในไทม์เฟรมต่างๆ
            timeframe_data = self._create_multi_timeframe_data(data)
            
            # วิเคราะห์ Elliott Wave ในแต่ละไทม์เฟรม
            for tf in self.timeframes:
                if tf in timeframe_data:
                    tf_analysis = self._analyze_elliott_wave_single_timeframe(
                        timeframe_data[tf], tf
                    )
                    results['timeframe_analysis'][tf] = tf_analysis
            
            # วิเคราะห์ Confluence ระหว่างไทม์เฟรม
            results['confluence_analysis'] = self._analyze_timeframe_confluence(
                results['timeframe_analysis']
            )
            
            # ระบุ Primary Wave Count
            results['primary_wave_count'] = self._determine_primary_wave_count(
                results['timeframe_analysis']
            )
            
            # คำนวณ Fibonacci Levels
            results['fibonacci_levels'] = self._calculate_advanced_fibonacci_levels(
                data, results['primary_wave_count']
            )
            
            # สร้าง Trading Signals
            results['trading_signals'] = self._generate_elliott_wave_signals(
                results
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Elliott Wave multi-timeframe analysis failed: {str(e)}")
            return self._get_fallback_elliott_analysis()
    
    def _create_multi_timeframe_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """สร้างข้อมูลในไทม์เฟรมต่างๆ"""
        timeframe_data = {}
        
        try:
            # ตรวจสอบว่าข้อมูลมี datetime column
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'])
            elif 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
            else:
                # ใช้ index เป็น datetime
                data = data.reset_index()
                data['datetime'] = pd.date_range(
                    start='2020-01-01', periods=len(data), freq='1min'
                )
            
            data.set_index('datetime', inplace=True)
            data.sort_index(inplace=True)
            
            # สร้างข้อมูลในไทม์เฟรมต่างๆ
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
                    
                    if len(resampled) >= 100:  # ต้องมีข้อมูลเพียงพอ
                        timeframe_data[tf] = resampled
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create {tf} timeframe: {str(e)}")
                    continue
            
            return timeframe_data
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-timeframe data: {str(e)}")
            return {'M1': data}  # Fallback to original data
    
    def _analyze_elliott_wave_single_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """วิเคราะห์ Elliott Wave ในไทม์เฟรมเดียว"""
        try:
            analysis = {
                'timeframe': timeframe,
                'wave_count': None,
                'wave_type': None,
                'current_wave': None,
                'wave_progress': 0.0,
                'trend_direction': 'NEUTRAL',
                'wave_strength': 0,
                'pivot_points': [],
                'fibonacci_levels': {},
                'wave_targets': {},
                'confidence': 0.0
            }
            
            # หา Pivot Points
            pivot_points = self._find_elliott_wave_pivots(data)
            analysis['pivot_points'] = pivot_points
            
            if len(pivot_points) >= 5:
                # วิเคราะห์ Wave Pattern
                wave_analysis = self._analyze_wave_pattern(pivot_points, data)
                analysis.update(wave_analysis)
                
                # คำนวณ Fibonacci Levels
                analysis['fibonacci_levels'] = self._calculate_fibonacci_levels(
                    pivot_points, data
                )
                
                # ประเมิน Wave Targets
                analysis['wave_targets'] = self._calculate_wave_targets(
                    pivot_points, analysis['wave_count']
                )
                
                # คำนวณ Confidence
                analysis['confidence'] = self._calculate_wave_confidence(
                    analysis, data
                )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Single timeframe Elliott Wave analysis failed: {str(e)}")
            return {
                'timeframe': timeframe,
                'wave_count': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _find_elliott_wave_pivots(self, data: pd.DataFrame, 
                                  window: int = 10) -> List[Dict[str, Any]]:
        """หา Pivot Points สำหรับ Elliott Wave"""
        try:
            pivots = []
            
            # คำนวณ Local Highs และ Lows
            highs = data['high'].rolling(window=window*2+1, center=True).max()
            lows = data['low'].rolling(window=window*2+1, center=True).min()
            
            for i in range(window, len(data) - window):
                # Check for Local High
                if data['high'].iloc[i] == highs.iloc[i] and data['high'].iloc[i] > 0:
                    pivot = {
                        'type': 'HIGH',
                        'price': data['high'].iloc[i],
                        'index': i,
                        'datetime': data.index[i],
                        'volume': data['volume'].iloc[i] if 'volume' in data.columns else 0
                    }
                    pivots.append(pivot)
                
                # Check for Local Low
                elif data['low'].iloc[i] == lows.iloc[i] and data['low'].iloc[i] > 0:
                    pivot = {
                        'type': 'LOW',
                        'price': data['low'].iloc[i],
                        'index': i,
                        'datetime': data.index[i],
                        'volume': data['volume'].iloc[i] if 'volume' in data.columns else 0
                    }
                    pivots.append(pivot)
            
            # เรียงตาม datetime
            pivots.sort(key=lambda x: x['datetime'])
            
            # กรองเอาเฉพาะ pivots ที่สำคัญ
            filtered_pivots = self._filter_significant_pivots(pivots)
            
            return filtered_pivots
            
        except Exception as e:
            self.logger.error(f"Failed to find Elliott Wave pivots: {str(e)}")
            return []
    
    def _filter_significant_pivots(self, pivots: List[Dict[str, Any]], 
                                   min_change_percent: float = 0.5) -> List[Dict[str, Any]]:
        """กรอง Pivot Points ที่สำคัญ"""
        if len(pivots) < 2:
            return pivots
        
        filtered = [pivots[0]]  # เริ่มด้วย pivot แรก
        
        for pivot in pivots[1:]:
            last_pivot = filtered[-1]
            
            # คำนวณ % การเปลี่ยนแปลง
            price_change = abs(pivot['price'] - last_pivot['price'])
            percent_change = (price_change / last_pivot['price']) * 100
            
            # เก็บเฉพาะ pivot ที่มีการเปลี่ยนแปลงมากพอ
            if percent_change >= min_change_percent:
                # ตรวจสอบว่าเป็น alternating high/low หรือไม่
                if pivot['type'] != last_pivot['type']:
                    filtered.append(pivot)
                else:
                    # ถ้าเป็น type เดียวกัน เลือกที่ extreme กว่า
                    if pivot['type'] == 'HIGH' and pivot['price'] > last_pivot['price']:
                        filtered[-1] = pivot
                    elif pivot['type'] == 'LOW' and pivot['price'] < last_pivot['price']:
                        filtered[-1] = pivot
        
        return filtered
    
    def _analyze_wave_pattern(self, pivots: List[Dict[str, Any]], 
                             data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบ Elliott Wave"""
        try:
            if len(pivots) < 5:
                return {
                    'wave_count': None,
                    'wave_type': None,
                    'current_wave': None,
                    'wave_progress': 0.0,
                    'trend_direction': 'NEUTRAL',
                    'wave_strength': 0
                }
            
            # วิเคราะห์ 5 waves ล่าสุด
            recent_pivots = pivots[-6:] if len(pivots) >= 6 else pivots
            
            # ระบุ Wave Pattern
            wave_pattern = self._identify_wave_pattern(recent_pivots)
            
            # ระบุ Current Wave
            current_wave = self._identify_current_wave(recent_pivots, data)
            
            # คำนวณ Wave Progress
            wave_progress = self._calculate_wave_progress(recent_pivots, current_wave)
            
            # ระบุ Trend Direction
            trend_direction = self._determine_trend_direction(recent_pivots)
            
            # คำนวณ Wave Strength
            wave_strength = self._calculate_wave_strength(recent_pivots, data)
            
            return {
                'wave_count': wave_pattern['count'],
                'wave_type': wave_pattern['type'],
                'current_wave': current_wave,
                'wave_progress': wave_progress,
                'trend_direction': trend_direction,
                'wave_strength': wave_strength,
                'pattern_details': wave_pattern
            }
            
        except Exception as e:
            self.logger.error(f"Wave pattern analysis failed: {str(e)}")
            return {
                'wave_count': None,
                'wave_type': 'UNKNOWN',
                'current_wave': None,
                'wave_progress': 0.0,
                'trend_direction': 'NEUTRAL',
                'wave_strength': 0
            }
    
    def _identify_wave_pattern(self, pivots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ระบุรูปแบบ Elliott Wave"""
        try:
            if len(pivots) < 5:
                return {'count': None, 'type': 'INCOMPLETE', 'confidence': 0.0}
            
            # วิเคราะห์ 5-wave impulse pattern
            impulse_score = self._analyze_impulse_pattern(pivots[-6:] if len(pivots) >= 6 else pivots)
            
            # วิเคราะห์ 3-wave corrective pattern  
            corrective_score = self._analyze_corrective_pattern(pivots[-4:] if len(pivots) >= 4 else pivots)
            
            # เลือก pattern ที่มี score สูงสุด
            if impulse_score['confidence'] > corrective_score['confidence']:
                return {
                    'count': '5-WAVE',
                    'type': 'IMPULSE',
                    'confidence': impulse_score['confidence'],
                    'details': impulse_score
                }
            else:
                return {
                    'count': '3-WAVE',
                    'type': 'CORRECTIVE',
                    'confidence': corrective_score['confidence'],
                    'details': corrective_score
                }
                
        except Exception as e:
            self.logger.error(f"Wave pattern identification failed: {str(e)}")
            return {'count': None, 'type': 'UNKNOWN', 'confidence': 0.0}
    
    def _analyze_impulse_pattern(self, pivots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ 5-wave impulse pattern"""
        try:
            if len(pivots) < 5:
                return {'confidence': 0.0, 'violations': ['Insufficient pivots']}
            
            confidence = 0.0
            violations = []
            wave_measurements = {}
            
            # ตรวจสอบ Elliott Wave Rules
            # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
            if len(pivots) >= 3:
                wave1_length = abs(pivots[1]['price'] - pivots[0]['price'])
                wave2_retrace = abs(pivots[2]['price'] - pivots[1]['price'])
                
                if wave2_retrace <= wave1_length:
                    confidence += 20
                    wave_measurements['wave2_retrace_valid'] = True
                else:
                    violations.append("Wave 2 retraces more than 100% of Wave 1")
                    wave_measurements['wave2_retrace_valid'] = False
            
            # Rule 2: Wave 3 is never the shortest wave
            if len(pivots) >= 5:
                wave1_length = abs(pivots[1]['price'] - pivots[0]['price'])
                wave3_length = abs(pivots[3]['price'] - pivots[2]['price'])
                wave5_length = abs(pivots[4]['price'] - pivots[3]['price']) if len(pivots) > 4 else 0
                
                if wave3_length >= wave1_length and (wave5_length == 0 or wave3_length >= wave5_length):
                    confidence += 30
                    wave_measurements['wave3_longest'] = True
                else:
                    violations.append("Wave 3 is the shortest wave")
                    wave_measurements['wave3_longest'] = False
            
            # Rule 3: Wave 4 cannot overlap Wave 1 price territory (in impulse waves)
            if len(pivots) >= 5:
                wave1_high = max(pivots[0]['price'], pivots[1]['price'])
                wave1_low = min(pivots[0]['price'], pivots[1]['price'])
                wave4_high = max(pivots[3]['price'], pivots[4]['price'])
                wave4_low = min(pivots[3]['price'], pivots[4]['price'])
                
                # Check for overlap
                overlap = not (wave4_high < wave1_low or wave4_low > wave1_high)
                if not overlap:
                    confidence += 25
                    wave_measurements['wave4_no_overlap'] = True
                else:
                    violations.append("Wave 4 overlaps Wave 1 territory")
                    wave_measurements['wave4_no_overlap'] = False
            
            # Fibonacci relationships
            fibonacci_score = self._check_fibonacci_relationships(pivots)
            confidence += fibonacci_score * 25
            
            return {
                'confidence': min(confidence, 100.0),
                'violations': violations,
                'wave_measurements': wave_measurements,
                'fibonacci_score': fibonacci_score
            }
            
        except Exception as e:
            self.logger.error(f"Impulse pattern analysis failed: {str(e)}")
            return {'confidence': 0.0, 'violations': ['Analysis failed']}
    
    def _analyze_corrective_pattern(self, pivots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ 3-wave corrective pattern"""
        try:
            if len(pivots) < 3:
                return {'confidence': 0.0, 'violations': ['Insufficient pivots']}
            
            confidence = 0.0
            violations = []
            
            # ใช้ 3 pivots ล่าสุดสำหรับ ABC pattern
            abc_pivots = pivots[-3:]
            
            # วิเคราะห์ ABC corrective pattern
            # Wave A และ Wave C ควรมีขนาดใกล้เคียงกัน
            wave_a_length = abs(abc_pivots[1]['price'] - abc_pivots[0]['price'])
            wave_c_length = abs(abc_pivots[2]['price'] - abc_pivots[1]['price'])
            
            # คำนวณ ratio ระหว่าง Wave A และ C
            if wave_a_length > 0:
                ac_ratio = wave_c_length / wave_a_length
                
                # Wave C should be 0.618 to 1.618 times Wave A
                if 0.618 <= ac_ratio <= 1.618:
                    confidence += 40
                else:
                    violations.append(f"Wave A/C ratio {ac_ratio:.3f} outside expected range")
            
            # Wave B retracement should be 38.2% to 78.6% of Wave A
            wave_b_retrace = abs(abc_pivots[1]['price'] - abc_pivots[2]['price'])
            if wave_a_length > 0:
                b_retrace_ratio = wave_b_retrace / wave_a_length
                
                if 0.382 <= b_retrace_ratio <= 0.786:
                    confidence += 30
                else:
                    violations.append(f"Wave B retracement {b_retrace_ratio:.3f} outside expected range")
            
            # Time relationships
            time_score = self._analyze_corrective_time_relationships(abc_pivots)
            confidence += time_score * 30
            
            return {
                'confidence': min(confidence, 100.0),
                'violations': violations,
                'ac_ratio': ac_ratio if 'ac_ratio' in locals() else 0,
                'b_retrace_ratio': b_retrace_ratio if 'b_retrace_ratio' in locals() else 0,
                'time_score': time_score
            }
            
        except Exception as e:
            self.logger.error(f"Corrective pattern analysis failed: {str(e)}")
            return {'confidence': 0.0, 'violations': ['Analysis failed']}
    
    def _check_fibonacci_relationships(self, pivots: List[Dict[str, Any]]) -> float:
        """ตรวจสอบความสัมพันธ์ Fibonacci ระหว่าง waves"""
        try:
            if len(pivots) < 3:
                return 0.0
            
            score = 0.0
            
            # ตรวจสอบ Fibonacci ratios ระหว่าง waves
            for i in range(len(pivots) - 2):
                wave1 = abs(pivots[i+1]['price'] - pivots[i]['price'])
                wave2 = abs(pivots[i+2]['price'] - pivots[i+1]['price'])
                
                if wave1 > 0:
                    ratio = wave2 / wave1
                    
                    # ตรวจสอบกับ Fibonacci ratios ที่สำคัญ
                    for fib_ratio in self.fibonacci_ratios['extensions']:
                        if abs(ratio - fib_ratio) / fib_ratio < 0.1:  # 10% tolerance
                            score += 0.2
                            break
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Fibonacci relationship check failed: {str(e)}")
            return 0.0
    
    def _analyze_corrective_time_relationships(self, pivots: List[Dict[str, Any]]) -> float:
        """วิเคราะห์ความสัมพันธ์ด้านเวลาใน corrective patterns"""
        try:
            if len(pivots) < 3:
                return 0.0
            
            # คำนวณระยะเวลาของแต่ละ wave
            time_a = (pivots[1]['datetime'] - pivots[0]['datetime']).total_seconds()
            time_c = (pivots[2]['datetime'] - pivots[1]['datetime']).total_seconds()
            
            if time_a > 0:
                time_ratio = time_c / time_a
                
                # Wave C ควรใช้เวลาใกล้เคียงกับ Wave A
                if 0.5 <= time_ratio <= 2.0:
                    return 1.0
                elif 0.3 <= time_ratio <= 3.0:
                    return 0.5
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Time relationship analysis failed: {str(e)}")
            return 0.0
    
    def _identify_current_wave(self, pivots: List[Dict[str, Any]], 
                              data: pd.DataFrame) -> Optional[str]:
        """ระบุ wave ปัจจุบัน"""
        try:
            if len(pivots) < 2:
                return None
            
            # ราคาปัจจุบัน
            current_price = data['close'].iloc[-1]
            last_pivot = pivots[-1]
            
            # ทิศทางการเคลื่อนไหวปัจจุบัน
            if current_price > last_pivot['price']:
                direction = 'UP'
            else:
                direction = 'DOWN'
            
            # ระบุ wave ตามจำนวน pivots และทิศทาง
            wave_count = len(pivots)
            
            if wave_count >= 5:
                # มี 5 pivots แล้ว อาจเป็น wave 5 หรือเริ่ม corrective
                if direction == 'UP' and pivots[-1]['type'] == 'LOW':
                    return 'WAVE_5' if wave_count % 2 == 1 else 'WAVE_A'
                elif direction == 'DOWN' and pivots[-1]['type'] == 'HIGH':
                    return 'WAVE_5' if wave_count % 2 == 0 else 'WAVE_A'
            elif wave_count == 4:
                return 'WAVE_5'
            elif wave_count == 3:
                return 'WAVE_4'
            elif wave_count == 2:
                return 'WAVE_3'
            elif wave_count == 1:
                return 'WAVE_2'
            
            return 'WAVE_1'
            
        except Exception as e:
            self.logger.error(f"Current wave identification failed: {str(e)}")
            return None
    
    def _calculate_wave_progress(self, pivots: List[Dict[str, Any]], 
                                current_wave: Optional[str]) -> float:
        """คำนวณความคืบหน้าของ wave ปัจจุบัน"""
        try:
            if not current_wave or len(pivots) < 2:
                return 0.0
            
            # คำนวณตาม Fibonacci projections
            if len(pivots) >= 2:
                last_pivot = pivots[-1]
                prev_pivot = pivots[-2]
                
                wave_length = abs(last_pivot['price'] - prev_pivot['price'])
                
                # ประมาณความคืบหน้าตาม wave type
                if 'WAVE_3' in current_wave:
                    return 0.6  # Wave 3 มักจะยาวกว่า wave อื่น
                elif 'WAVE_5' in current_wave:
                    return 0.8  # Wave 5 ใกล้จบ impulse
                elif 'WAVE_A' in current_wave:
                    return 0.3  # เริ่ม correction
                elif 'WAVE_C' in current_wave:
                    return 0.7  # ใกล้จบ correction
                else:
                    return 0.5  # Default
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Wave progress calculation failed: {str(e)}")
            return 0.0
    
    def _determine_trend_direction(self, pivots: List[Dict[str, Any]]) -> str:
        """ระบุทิศทางเทรนด์หลัก"""
        try:
            if len(pivots) < 3:
                return 'NEUTRAL'
            
            # เปรียบเทียบ pivot highs และ lows
            recent_pivots = pivots[-3:]
            
            highs = [p['price'] for p in recent_pivots if p['type'] == 'HIGH']
            lows = [p['price'] for p in recent_pivots if p['type'] == 'LOW']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Higher highs และ higher lows = uptrend
                if highs[-1] > highs[0] and lows[-1] > lows[0]:
                    return 'UPTREND'
                # Lower highs และ lower lows = downtrend
                elif highs[-1] < highs[0] and lows[-1] < lows[0]:
                    return 'DOWNTREND'
            
            return 'SIDEWAYS'
            
        except Exception as e:
            self.logger.error(f"Trend direction determination failed: {str(e)}")
            return 'NEUTRAL'
    
    def _calculate_wave_strength(self, pivots: List[Dict[str, Any]], 
                                data: pd.DataFrame) -> int:
        """คำนวณความแข็งแกร่งของ wave (0-10)"""
        try:
            if len(pivots) < 2:
                return 0
            
            strength = 0
            
            # Volume confirmation
            if 'volume' in data.columns:
                recent_volume = data['volume'].tail(20).mean()
                avg_volume = data['volume'].mean()
                
                if recent_volume > avg_volume * 1.2:
                    strength += 3
                elif recent_volume > avg_volume:
                    strength += 2
            
            # Price momentum
            price_change = abs(pivots[-1]['price'] - pivots[-2]['price'])
            avg_price = (pivots[-1]['price'] + pivots[-2]['price']) / 2
            momentum = (price_change / avg_price) * 100
            
            if momentum > 2.0:
                strength += 3
            elif momentum > 1.0:
                strength += 2
            elif momentum > 0.5:
                strength += 1
            
            # Time factor
            time_diff = (pivots[-1]['datetime'] - pivots[-2]['datetime']).total_seconds()
            if time_diff < 3600:  # Less than 1 hour = strong momentum
                strength += 2
            elif time_diff < 7200:  # Less than 2 hours
                strength += 1
            
            # Wave count consistency
            if len(pivots) >= 5:
                strength += 2
            
            return min(strength, 10)
            
        except Exception as e:
            self.logger.error(f"Wave strength calculation failed: {str(e)}")
            return 0
    
    def _analyze_timeframe_confluence(self, timeframe_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """วิเคราะห์ Confluence ระหว่างไทม์เฟรม"""
        try:
            confluence = {
                'overall_direction': 'NEUTRAL',
                'strength': 0,
                'agreement_score': 0.0,
                'conflicting_timeframes': [],
                'supporting_timeframes': [],
                'primary_timeframe': None
            }
            
            # นับการโหวตทิศทาง
            direction_votes = {'UPTREND': 0, 'DOWNTREND': 0, 'SIDEWAYS': 0, 'NEUTRAL': 0}
            valid_timeframes = []
            
            for tf, analysis in timeframe_analysis.items():
                if analysis.get('confidence', 0) > 30:  # ใช้เฉพาะ timeframe ที่มั่นใจ
                    direction = analysis.get('trend_direction', 'NEUTRAL')
                    direction_votes[direction] += 1
                    valid_timeframes.append(tf)
            
            # หาทิศทางที่ได้รับการสนับสนุนมากที่สุด
            if valid_timeframes:
                primary_direction = max(direction_votes, key=direction_votes.get)
                confluence['overall_direction'] = primary_direction
                
                # คำนวณ agreement score
                total_votes = sum(direction_votes.values())
                agreement_score = direction_votes[primary_direction] / total_votes if total_votes > 0 else 0
                confluence['agreement_score'] = agreement_score
                
                # ระบุ supporting และ conflicting timeframes
                for tf, analysis in timeframe_analysis.items():
                    if analysis.get('confidence', 0) > 30:
                        tf_direction = analysis.get('trend_direction', 'NEUTRAL')
                        if tf_direction == primary_direction:
                            confluence['supporting_timeframes'].append(tf)
                        else:
                            confluence['conflicting_timeframes'].append(tf)
                
                # คำนวณ overall strength
                confluence['strength'] = int(agreement_score * 10)
                
                # ระบุ primary timeframe (timeframe ที่มี confidence สูงสุด)
                best_tf = max(timeframe_analysis.keys(), 
                            key=lambda x: timeframe_analysis[x].get('confidence', 0))
                confluence['primary_timeframe'] = best_tf
            
            return confluence
            
        except Exception as e:
            self.logger.error(f"Timeframe confluence analysis failed: {str(e)}")
            return {
                'overall_direction': 'NEUTRAL',
                'strength': 0,
                'agreement_score': 0.0,
                'conflicting_timeframes': [],
                'supporting_timeframes': [],
                'primary_timeframe': None
            }
    
    def _determine_primary_wave_count(self, timeframe_analysis: Dict[str, Any]) -> Optional[str]:
        """ระบุ Primary Wave Count จากการวิเคราะห์หลายไทม์เฟรม"""
        try:
            # หา timeframe ที่มี confidence สูงสุด
            best_timeframe = None
            best_confidence = 0
            
            for tf, analysis in timeframe_analysis.items():
                confidence = analysis.get('confidence', 0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_timeframe = tf
            
            if best_timeframe and best_confidence > 50:
                return timeframe_analysis[best_timeframe].get('wave_count')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Primary wave count determination failed: {str(e)}")
            return None
    
    def _calculate_advanced_fibonacci_levels(self, data: pd.DataFrame, 
                                           primary_wave_count: Optional[str]) -> Dict[str, Any]:
        """คำนวณ Fibonacci Levels ขั้นสูง"""
        try:
            fibonacci_levels = {
                'retracements': {},
                'extensions': {},
                'projections': {},
                'key_levels': []
            }
            
            if len(data) < 50:
                return fibonacci_levels
            
            # หา swing high และ low สำคัญ
            recent_data = data.tail(100)
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            current_price = recent_data['close'].iloc[-1]
            
            # คำนวณ Fibonacci Retracements
            price_range = swing_high - swing_low
            for ratio in self.fibonacci_ratios['retracements']:
                if current_price < swing_high:  # ในการ retrace จาก high
                    level = swing_high - (price_range * ratio)
                else:  # ในการ retrace จาก low
                    level = swing_low + (price_range * ratio)
                
                fibonacci_levels['retracements'][f'{ratio:.1%}'] = level
                
                # ระบุ key levels ใกล้ราคาปัจจุบัน
                if abs(level - current_price) / current_price < 0.02:  # ภายใน 2%
                    fibonacci_levels['key_levels'].append({
                        'type': 'retracement',
                        'level': level,
                        'ratio': ratio,
                        'distance_percent': abs(level - current_price) / current_price * 100
                    })
            
            # คำนวณ Fibonacci Extensions
            for ratio in self.fibonacci_ratios['extensions']:
                if swing_high > swing_low:  # Uptrend
                    extension = swing_high + (price_range * ratio)
                else:  # Downtrend
                    extension = swing_low - (price_range * ratio)
                
                fibonacci_levels['extensions'][f'{ratio:.1%}'] = extension
            
            # คำนวณ Fibonacci Projections สำหรับ Wave targets
            if primary_wave_count:
                projections = self._calculate_wave_projections(
                    data, primary_wave_count, swing_high, swing_low
                )
                fibonacci_levels['projections'] = projections
            
            return fibonacci_levels
            
        except Exception as e:
            self.logger.error(f"Advanced Fibonacci calculation failed: {str(e)}")
            return {
                'retracements': {},
                'extensions': {},
                'projections': {},
                'key_levels': []
            }
    
    def _calculate_wave_projections(self, data: pd.DataFrame, wave_count: str,
                                   swing_high: float, swing_low: float) -> Dict[str, float]:
        """คำนวณ Wave Projections ตาม Elliott Wave theory"""
        try:
            projections = {}
            price_range = abs(swing_high - swing_low)
            
            if '5-WAVE' in wave_count:
                # Wave 3 target: 1.618 * Wave 1
                projections['wave_3_target'] = swing_low + (price_range * 1.618)
                
                # Wave 5 target: equal to Wave 1 or 0.618 * Wave 1-3
                projections['wave_5_target_equal'] = swing_high + price_range
                projections['wave_5_target_618'] = swing_high + (price_range * 0.618)
                
            elif '3-WAVE' in wave_count:
                # Wave C target: equal to Wave A or 1.618 * Wave A
                projections['wave_c_target_equal'] = swing_low - price_range
                projections['wave_c_target_618'] = swing_low - (price_range * 1.618)
            
            return projections
            
        except Exception as e:
            self.logger.error(f"Wave projections calculation failed: {str(e)}")
            return {}
    
    def _generate_elliott_wave_signals(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """สร้าง Trading Signals จากการวิเคราะห์ Elliott Wave"""
        try:
            signals = []
            
            confluence = analysis_results.get('confluence_analysis', {})
            overall_direction = confluence.get('overall_direction', 'NEUTRAL')
            strength = confluence.get('strength', 0)
            primary_wave = analysis_results.get('primary_wave_count')
            
            # Signal 1: Trend Following
            if strength >= 7:  # Strong confluence
                if overall_direction == 'UPTREND':
                    signals.append({
                        'type': 'BUY',
                        'strength': strength,
                        'reason': 'Strong uptrend confluence across timeframes',
                        'confidence': confluence.get('agreement_score', 0) * 100,
                        'timeframes_supporting': len(confluence.get('supporting_timeframes', []))
                    })
                elif overall_direction == 'DOWNTREND':
                    signals.append({
                        'type': 'SELL',
                        'strength': strength,
                        'reason': 'Strong downtrend confluence across timeframes',
                        'confidence': confluence.get('agreement_score', 0) * 100,
                        'timeframes_supporting': len(confluence.get('supporting_timeframes', []))
                    })
            
            # Signal 2: Wave Completion Signals
            if primary_wave:
                if '5-WAVE' in primary_wave:
                    # Wave 5 completion - prepare for reversal
                    signals.append({
                        'type': 'REVERSAL_WARNING',
                        'strength': 6,
                        'reason': 'Wave 5 near completion - expect corrective move',
                        'confidence': 70,
                        'wave_type': 'impulse_completion'
                    })
                elif 'WAVE_C' in primary_wave:
                    # Wave C completion - prepare for new impulse
                    signals.append({
                        'type': 'TREND_RESUMPTION',
                        'strength': 7,
                        'reason': 'Wave C correction near completion',
                        'confidence': 75,
                        'wave_type': 'correction_completion'
                    })
            
            # Signal 3: Fibonacci Level Signals
            fibonacci_levels = analysis_results.get('fibonacci_levels', {})
            key_levels = fibonacci_levels.get('key_levels', [])
            
            for level in key_levels:
                if level['distance_percent'] < 1.0:  # ใกล้ Fibonacci level
                    signals.append({
                        'type': 'FIBONACCI_LEVEL',
                        'strength': 5,
                        'reason': f"Price near {level['ratio']:.1%} Fibonacci {level['type']}",
                        'confidence': 60,
                        'level_price': level['level'],
                        'fibonacci_ratio': level['ratio']
                    })
            
            # เรียงตาม strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:5]  # Return top 5 signals
            
        except Exception as e:
            self.logger.error(f"Elliott Wave signal generation failed: {str(e)}")
            return []
    
    def _calculate_fibonacci_levels(self, pivots: List[Dict[str, Any]], 
                                   data: pd.DataFrame) -> Dict[str, float]:
        """คำนวณ Fibonacci levels จาก pivot points"""
        try:
            if len(pivots) < 2:
                return {}
            
            # ใช้ 2 pivots ล่าสุด
            high_pivot = max(pivots[-2:], key=lambda x: x['price'])
            low_pivot = min(pivots[-2:], key=lambda x: x['price'])
            
            price_range = high_pivot['price'] - low_pivot['price']
            
            fibonacci_levels = {}
            
            # Retracement levels
            for ratio in self.fibonacci_ratios['retracements']:
                level = high_pivot['price'] - (price_range * ratio)
                fibonacci_levels[f'ret_{ratio:.1%}'] = level
            
            # Extension levels
            for ratio in self.fibonacci_ratios['extensions']:
                level = high_pivot['price'] + (price_range * ratio)
                fibonacci_levels[f'ext_{ratio:.1%}'] = level
            
            return fibonacci_levels
            
        except Exception as e:
            self.logger.error(f"Fibonacci levels calculation failed: {str(e)}")
            return {}
    
    def _calculate_wave_targets(self, pivots: List[Dict[str, Any]], 
                               wave_count: Optional[str]) -> Dict[str, float]:
        """คำนวณ Wave targets"""
        try:
            if not wave_count or len(pivots) < 3:
                return {}
            
            targets = {}
            
            # คำนวณตาม wave type
            if '5-WAVE' in wave_count:
                # Wave 3 และ Wave 5 targets
                wave_1_length = abs(pivots[1]['price'] - pivots[0]['price'])
                
                # Wave 3 target (1.618 * Wave 1)
                if pivots[1]['price'] > pivots[0]['price']:  # Uptrend
                    targets['wave_3_target'] = pivots[1]['price'] + (wave_1_length * 1.618)
                    targets['wave_5_target'] = pivots[1]['price'] + wave_1_length
                else:  # Downtrend
                    targets['wave_3_target'] = pivots[1]['price'] - (wave_1_length * 1.618)
                    targets['wave_5_target'] = pivots[1]['price'] - wave_1_length
            
            elif '3-WAVE' in wave_count:
                # Wave C target
                wave_a_length = abs(pivots[1]['price'] - pivots[0]['price'])
                
                if pivots[1]['price'] > pivots[0]['price']:  # Correction in uptrend
                    targets['wave_c_target'] = pivots[2]['price'] - wave_a_length
                else:  # Correction in downtrend
                    targets['wave_c_target'] = pivots[2]['price'] + wave_a_length
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Wave targets calculation failed: {str(e)}")
            return {}
    
    def _calculate_wave_confidence(self, analysis: Dict[str, Any], 
                                  data: pd.DataFrame) -> float:
        """คำนวณความมั่นใจในการวิเคราะห์ Elliott Wave"""
        try:
            confidence = 0.0
            
            # Pattern quality score
            if analysis.get('pattern_details'):
                pattern_confidence = analysis['pattern_details'].get('confidence', 0)
                confidence += pattern_confidence * 0.4
            
            # Volume confirmation
            if 'volume' in data.columns and len(data) > 20:
                recent_volume = data['volume'].tail(10).mean()
                avg_volume = data['volume'].tail(50).mean()
                
                if recent_volume > avg_volume:
                    confidence += 20
            
            # Trend consistency
            if analysis.get('trend_direction') in ['UPTREND', 'DOWNTREND']:
                confidence += 15
            
            # Wave strength
            wave_strength = analysis.get('wave_strength', 0)
            confidence += wave_strength * 2
            
            # Fibonacci level proximity
            if analysis.get('fibonacci_levels'):
                confidence += 10
            
            return min(confidence, 100.0)
            
        except Exception as e:
            self.logger.error(f"Wave confidence calculation failed: {str(e)}")
            return 0.0
    
    def _get_fallback_elliott_analysis(self) -> Dict[str, Any]:
        """Fallback Elliott Wave analysis"""
        return {
            'timeframe_analysis': {},
            'confluence_analysis': {
                'overall_direction': 'NEUTRAL',
                'strength': 0,
                'agreement_score': 0.0
            },
            'primary_wave_count': None,
            'trend_direction': 'NEUTRAL',
            'wave_strength': 0,
            'fibonacci_levels': {},
            'trading_signals': [],
            'fallback': True,
            'error': 'Analysis failed - using fallback'
        }

# Export class
__all__ = ['AdvancedMultiTimeframeElliottWaveAnalyzer']
