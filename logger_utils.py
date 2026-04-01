"""
Centralized logging utility for YOLOv8 Object Detection project
Provides consistent logging configuration across all modules
"""

import os
import logging
from datetime import datetime
import config


def setup_logger(name, log_filename=None):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Name of the logger (typically __name__ from calling module)
        log_filename: Optional custom log filename. If None, uses the module name.
                     Can include path or just filename.
    
    Returns:
        Configured logger instance
    """
    # Ensure version-specific stats directory exists
    os.makedirs(config.VERSION_STATS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_filename is None:
        # Extract module name from logger name
        module_name = name.split('.')[-1] if '.' in name else name
        log_filename = f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Ensure log file is in version-specific stats directory
    if not os.path.dirname(log_filename):
        log_filename = os.path.join(config.VERSION_STATS_DIR, log_filename)
    
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Capture more detail in file
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Get an existing logger or create a new one if it doesn't exist.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger
