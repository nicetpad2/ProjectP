#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ MENU SYSTEM
Enterprise Menu System for NICEGOLD ProjectP

This module provides the main menu system functionality.
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import unified master menu system
try:
    from .unified_master_menu_system import UnifiedMasterMenuSystem
    UNIFIED_MENU_AVAILABLE = True
except ImportError:
    UNIFIED_MENU_AVAILABLE = False

class MenuSystem:
    """
    Main Menu System for NICEGOLD ProjectP
    
    Provides compatibility wrapper for unified menu system
    """
    
    def __init__(self):
        """Initialize Menu System"""
        self.unified_menu = None
        
        # Initialize unified menu system if available
        if UNIFIED_MENU_AVAILABLE:
            try:
                from .unified_master_menu_system import UnifiedMasterMenuSystem
                self.unified_menu = UnifiedMasterMenuSystem()
            except Exception:
                self.unified_menu = None
    
    def initialize(self):
        """Initialize the menu system"""
        if self.unified_menu:
            return self.unified_menu.initialize_components()
        return True
    
    def display_menu(self):
        """Display the main menu"""
        if self.unified_menu:
            return self.unified_menu.display_unified_menu()
        else:
            # Fallback basic menu
            print("\n🏢 NICEGOLD ENTERPRISE PROJECTP - MENU SYSTEM")
            print("=" * 60)
            print("1. 🌊 Elliott Wave Full Pipeline")
            print("2. 📊 System Status")
            print("3. 🔧 Configuration")
            print("E. 🚪 Exit")
            print("=" * 60)
    
    def handle_selection(self, choice: str):
        """Handle user menu selection"""
        if self.unified_menu:
            return self.unified_menu.handle_menu_choice(choice)
        return choice
    
    def run(self):
        """Run the menu system"""
        if self.unified_menu:
            return self.unified_menu.start()
        else:
            # Basic fallback menu loop
            while True:
                self.display_menu()
                choice = input("\nEnter your choice: ").strip().upper()
                
                if choice == 'E':
                    print("👋 Thank you for using NICEGOLD ProjectP!")
                    break
                elif choice == '1':
                    print("🌊 Elliott Wave Pipeline would run here...")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

# Factory function for compatibility
def create_menu_system():
    """Create menu system instance"""
    return MenuSystem()

def get_menu_system():
    """Get menu system instance (main factory function)"""
    return MenuSystem()

# Alias for backward compatibility
MainMenuSystem = MenuSystem

__all__ = [
    'MenuSystem',
    'MainMenuSystem',
    'create_menu_system',
    'get_menu_system'
] 