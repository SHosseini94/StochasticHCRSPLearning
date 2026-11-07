"""
Logging setup module for EVT-based project.
Configures root logger to write to both console and rotating file.
"""
import logging
from logging.handlers import RotatingFileHandler
import os


def configure_logging(base_dir: str) -> None:
    """Configure root logger to log to both console and rotating file.

    Args:
        base_dir: Base directory (e.g., EVT-based folder)

    Creates logs at: base_dir/result/logs/app.log
    - Rotating: 2MB max per file, keeps 5 backups
    - Console: INFO level and above
    """
    logs_dir = os.path.join(base_dir, 'result', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, 'app.log')

    root = logging.getLogger()
    # Prevent duplicate handlers if called multiple times
    if root.handlers:
        return

    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler: standard file handler (append mode)
    fh = logging.FileHandler(
        log_path, 
        mode='a',  # Append mode - preserves existing logs
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Console handler: stdout (WARNING level only - minimal output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # Only show warnings/errors on console
    ch.setFormatter(fmt)

    root.addHandler(fh)
    # root.addHandler(ch)  # Disabled: console output only goes to file now
