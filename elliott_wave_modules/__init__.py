#!/usr/bin/env python3
"""
Elliott Wave Modules Package
🌊 Enterprise-Grade Elliott Wave CNN-LSTM + DQN System
"""

__version__ = "2.0 DIVINE EDITION"
__author__ = "NICEGOLD Enterprise"

# Import with error handling
try:
    from .data_processor import ElliottWaveDataProcessor
    from .cnn_lstm_engine import CNNLSTMElliottWave  
    from .dqn_agent import DQNReinforcementAgent
    from .feature_selector import EnterpriseShapOptunaFeatureSelector, SHAPOptunaFeatureSelector
    from .pipeline_orchestrator import ElliottWavePipelineOrchestrator
    from .performance_analyzer import ElliottWavePerformanceAnalyzer
    
    __all__ = [
        'ElliottWaveDataProcessor',
        'CNNLSTMElliottWave',
        'DQNReinforcementAgent',
        'EnterpriseShapOptunaFeatureSelector',
        'SHAPOptunaFeatureSelector',
        'ElliottWavePipelineOrchestrator',
        'ElliottWavePerformanceAnalyzer'
    ]
    
except ImportError as e:
    print(f"⚠️ Warning: Some Elliott Wave modules could not be imported: {e}")
    __all__ = []
