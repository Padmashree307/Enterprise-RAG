import logging
import sys
from pathlib import Path

def setup_logger(name: str = "rag_system", log_file: str = "rag.log", level=logging.INFO):
    """
    Sets up a logger that writes to both file and console.
    """
    # Create logs directory if it doesn't exist
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times (if called repeatedly)
    if logger.hasHandlers():
        return logger
        
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File Handler
    file_handler = logging.FileHandler(log_path / log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Default logger instance
logger = setup_logger()
