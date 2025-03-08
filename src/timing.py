#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timing utilities for performance tracking.
"""

import time
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List
import logging
from datetime import datetime

from .logging_config import register_logger

# Register this module's logger
LOGGER_NAME = 'timing'
register_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)

class TimingManager:
    """Manages timing data collection and storage"""
    
    def __init__(self, enabled: bool = False, save_to_file: bool = False):
        """
        Initialize the timing manager.
        
        Args:
            enabled: Whether timing is enabled
            save_to_file: Whether to save timing data to file
        """
        self.enabled = enabled
        self.save_to_file = save_to_file
        self.timings: List[Dict] = []
        self.current_query: Optional[str] = None
        
        # Create timing directory if saving is enabled
        if save_to_file:
            self.timing_dir = Path("data/timing")
            self.timing_dir.mkdir(exist_ok=True, parents=True)
    
    def start_query(self, query: str):
        """Start timing a new query"""
        self.current_query = query
    
    def add_timing(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Add a timing entry"""
        if not self.enabled:
            return
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": self.current_query,
            "operation": operation,
            "duration_seconds": duration
        }
        if metadata:
            entry["metadata"] = metadata
            
        self.timings.append(entry)
        
        # Log the timing
        logger.info(f"Operation '{operation}' took {duration:.3f} seconds")
        
        # Save to file if enabled
        if self.save_to_file:
            self._save_timings()
    
    def _save_timings(self):
        """Save timing data to file"""
        if not self.save_to_file or not self.timings:
            return
            
        # Create a filename with timestamp
        filename = self.timing_dir / f"timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.timings, f, indent=2)
            
        logger.info(f"Saved timing data to {filename}")
    
    @contextmanager
    def timer(self, operation: str, metadata: Optional[Dict] = None):
        """Context manager for timing operations"""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.add_timing(operation, duration, metadata)

def get_timing_manager() -> TimingManager:
    """Get the timing manager instance"""
    if not hasattr(get_timing_manager, 'timing_manager'):
        get_timing_manager.timing_manager = TimingManager()
    return get_timing_manager.timing_manager

def initialize_timing(enabled: bool = False, save_to_file: bool = False):
    """Initialize the timing manager with specified settings"""
    get_timing_manager.timing_manager = TimingManager(enabled=enabled, save_to_file=save_to_file) 