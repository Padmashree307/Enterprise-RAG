import time
import functools
import logging
import os
import psutil

# Get the main logger
logger = logging.getLogger("rag_system")

def timer(func):
    """
    Decorator to measure execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
        return result
    return wrapper

def log_memory_usage(tag: str = ""):
    """
    Logs current memory usage of the process.
    Requires 'psutil' to be installed.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        logger.info(f"Memory Usage [{tag}]: {rss_mb:.2f} MB")
        return rss_mb
    except Exception as e:
        logger.warning(f"Failed to check memory usage: {e}")
        return 0.0
