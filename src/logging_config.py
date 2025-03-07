#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centralized logging configuration for the RAG-powered SQL Assistant.
"""

import logging
from typing import List, Optional, Set
from enum import Enum

class LogLevel(str, Enum):
    """Log level options"""
    NONE = "none"
    INFO = "info"
    DEBUG = "debug"

# Registry for logger names
_logger_names: Set[str] = set()

def register_logger(name: str) -> None:
    """
    Register a logger name with the logging configuration.
    
    Args:
        name: Name of the logger to register
    """
    _logger_names.add(name)

def get_all_loggers() -> List[str]:
    """
    Get all registered logger names.
    
    Returns:
        List of logger names
    """
    return sorted(_logger_names)

def setup_logging(log_level: LogLevel = LogLevel.NONE) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        log_level: Desired logging level
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    if log_level == LogLevel.NONE:
        # Set up a null handler to suppress all logging
        root_logger.addHandler(logging.NullHandler())
        root_logger.setLevel(logging.CRITICAL)  # Set to highest level to suppress all
        
        # Disable propagation for all loggers
        for logger_name in get_all_loggers():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
        return
    
    # Configure logging for the specified level
    level = logging.DEBUG if log_level == LogLevel.DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure all loggers
    for logger_name in get_all_loggers():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True 