#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 ENTERPRISE COMPONENT BASE CLASS
Base class for all enterprise components with standardized initialization

Features:
- Standardized component initialization
- Enterprise logging integration
- Health checking capabilities
- Error handling and recovery
- Component registry support
"""

import sys
import os
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Enterprise Components
from core.project_paths import get_project_paths

# Advanced Logging Integration
try:
    from core.unified_enterprise_logger import get_unified_logger
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False


class EnterpriseComponentBase:
    """Base class for all enterprise components"""
    
    def __init__(self, component_name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize enterprise component with standardized attributes
        
        Args:
            component_name: Name of the component (defaults to class name)
            config: Configuration dictionary
        """
        # Set component name
        self.component_name = component_name or self.__class__.__name__
        
        # Initialize standard attributes
        self.session_id = None
        self.start_time = datetime.now()
        self.metadata = {}
        self.status = "initializing"
        self.config = config or {}
        
        # Initialize component-specific attributes
        self.processing_stats = defaultdict(int)
        self.error_count = 0
        self.warning_count = 0
        self.success_count = 0
        
        # Setup enterprise logging
        self._setup_enterprise_logging()
        
        # Initialize project paths
        self.paths = get_project_paths()
        
        # Register component
        self.component_id = self._register_component()
        
        # Mark as initialized
        self.status = "ready"
        self.logger.info(f"✅ {self.component_name} initialized successfully")
    
    def _setup_enterprise_logging(self):
        """Setup enterprise logging system"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger = get_unified_logger()
                self.logger.info(f"🚀 {self.component_name} initialized with advanced logging")
            else:
                # Fallback to basic logging
                import logging
                self.logger = logging.getLogger(self.component_name)
                if not self.logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
                    self.logger.setLevel(logging.INFO)
        except Exception as e:
            print(f"⚠️ Enterprise logging setup failed for {self.component_name}: {e}")
            # Create minimal logger as last resort
            import logging
            self.logger = logging.getLogger(self.component_name)
    
    def _register_component(self) -> str:
        """Register component with enterprise system"""
        try:
            component_id = f"{self.component_name}_{uuid.uuid4().hex[:8]}"
            
            # Register with component registry if available
            if hasattr(self, '_component_registry'):
                self._component_registry.register(self)
            
            self.logger.info(f"📋 Component registered: {component_id}")
            return component_id
            
        except Exception as e:
            self.logger.warning(f"⚠️ Component registration failed: {e}")
            return f"{self.component_name}_fallback"
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on component"""
        try:
            health_status = {
                'component_name': self.component_name,
                'component_id': self.component_id,
                'status': self.status,
                'uptime': (datetime.now() - self.start_time).total_seconds(),
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'success_count': self.success_count,
                'last_check': datetime.now().isoformat(),
                'health_score': self._calculate_health_score()
            }
            
            return health_status
            
        except Exception as e:
            return {
                'component_name': self.component_name,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat(),
                'health_score': 0
            }
    
    def _calculate_health_score(self) -> float:
        """Calculate health score based on component performance"""
        try:
            total_operations = self.success_count + self.error_count + self.warning_count
            
            if total_operations == 0:
                return 1.0  # Perfect score for new component
            
            # Calculate weighted score
            success_weight = 1.0
            warning_weight = 0.5
            error_weight = 0.0
            
            weighted_score = (
                (self.success_count * success_weight) +
                (self.warning_count * warning_weight) +
                (self.error_count * error_weight)
            ) / total_operations
            
            return max(0.0, min(1.0, weighted_score))
            
        except Exception:
            return 0.0
    
    def log_success(self, message: str, metadata: Optional[Dict] = None):
        """Log successful operation"""
        self.success_count += 1
        if self.logger:
            self.logger.info(f"✅ {message}", extra=metadata or {})
        else:
            print(f"✅ {self.component_name}: {message}")
    
    def log_warning(self, message: str, metadata: Optional[Dict] = None):
        """Log warning"""
        self.warning_count += 1
        if self.logger:
            self.logger.warning(f"⚠️ {message}", extra=metadata or {})
        else:
            print(f"⚠️ {self.component_name}: {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None, metadata: Optional[Dict] = None):
        """Log error with optional exception details"""
        self.error_count += 1
        if self.logger:
            self.logger.error(f"❌ {message}", extra=metadata or {})
            if exception:
                self.logger.error(f"Exception details: {str(exception)}")
                self.logger.debug(traceback.format_exc())
        else:
            print(f"❌ {self.component_name}: {message}")
            if exception:
                print(f"Exception details: {str(exception)}")
                print(traceback.format_exc())
    
    def set_session_id(self, session_id: str):
        """Set session ID for tracking"""
        self.session_id = session_id
        self.logger.info(f"📋 Session ID set: {session_id}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata"""
        return {
            'component_name': self.component_name,
            'component_id': self.component_id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'status': self.status,
            'processing_stats': dict(self.processing_stats),
            'health_score': self._calculate_health_score()
        }
    
    def safe_execute(self, func, *args, **kwargs):
        """Safely execute function with error handling"""
        try:
            result = func(*args, **kwargs)
            self.log_success(f"Function {func.__name__} executed successfully")
            return result
        except Exception as e:
            self.log_error(f"Function {func.__name__} failed", e)
            raise
    
    def __str__(self):
        return f"{self.component_name}({self.component_id})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.component_name}({self.status})>"


class EnterpriseComponentRegistry:
    """Registry for all enterprise components"""
    
    _instance = None
    _components = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, component: EnterpriseComponentBase) -> str:
        """Register component with enterprise system"""
        try:
            component_id = component.component_id
            cls._components[component_id] = {
                'component': component,
                'registered_at': datetime.now(),
                'status': component.status,
                'health_check_count': 0
            }
            return component_id
        except Exception as e:
            print(f"⚠️ Component registration failed: {e}")
            return None
    
    @classmethod
    def get_component(cls, component_id: str) -> Optional[EnterpriseComponentBase]:
        """Get component by ID"""
        component_info = cls._components.get(component_id)
        return component_info['component'] if component_info else None
    
    @classmethod
    def health_check_all(cls) -> Dict[str, Any]:
        """Perform health check on all registered components"""
        health_report = {}
        
        for comp_id, comp_info in cls._components.items():
            try:
                component = comp_info['component']
                health_status = component.health_check()
                health_report[comp_id] = health_status
                comp_info['health_check_count'] += 1
            except Exception as e:
                health_report[comp_id] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
        
        return health_report
    
    @classmethod
    def get_all_components(cls) -> Dict[str, Any]:
        """Get all registered components"""
        return cls._components.copy()
    
    @classmethod
    def unregister(cls, component_id: str) -> bool:
        """Unregister component"""
        try:
            if component_id in cls._components:
                del cls._components[component_id]
                return True
            return False
        except Exception:
            return False


# Global registry instance
_component_registry = EnterpriseComponentRegistry()


def get_component_registry() -> EnterpriseComponentRegistry:
    """Get global component registry"""
    return _component_registry


# Export main classes and functions
__all__ = [
    'EnterpriseComponentBase',
    'EnterpriseComponentRegistry',
    'get_component_registry'
]
