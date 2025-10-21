import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "llm_router", level: str = "INFO") -> logging.Logger:
    """
    Simplified logging system setup for the project
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid setting up logger multiple times
    if logger.handlers:
        return logger
    
    # Set log level
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    logger.setLevel(log_levels.get(level.upper(), logging.INFO))
    
    # Create logs directory
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Log format
    log_format = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Colored console format
    class ColoredFormatter(logging.Formatter):
        """Colored log formatter"""
        
        COLORS = {
            'DEBUG': '\033[36m',     # Light blue
            'INFO': '\033[32m',      # Green
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[41m'   # Red background
        }
        RESET = '\033[0m'
        
        def format(self, record):
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
    
    colored_format = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colored_format)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Log session start
    logger.info(f"Starting logger - {name}")
    
    return logger


# Test example
if __name__ == "__main__":
    logger = setup_logger("test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    print("Logger test completed successfully")