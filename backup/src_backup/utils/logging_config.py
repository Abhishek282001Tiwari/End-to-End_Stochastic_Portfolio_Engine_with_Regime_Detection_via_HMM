import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    logger.remove()
    
    if log_format is None:
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format,
        colorize=True
    )
    
    if log_file:
        log_path = Path("logs") / log_file
        log_path.parent.mkdir(exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format=log_format,
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str):
    return logger.bind(name=name)