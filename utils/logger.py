"""
================================================================
  utils/logger.py
  Centralised logging setup for the entire system.
  Outputs to both console (coloured) and rotating log file.
================================================================
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── ANSI colour codes for terminal output ────────────────────
class Colours:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"


class ColouredFormatter(logging.Formatter):
    LEVEL_COLOURS = {
        logging.DEBUG:    Colours.GREY,
        logging.INFO:     Colours.CYAN,
        logging.WARNING:  Colours.YELLOW,
        logging.ERROR:    Colours.RED,
        logging.CRITICAL: Colours.RED + Colours.BOLD,
    }

    def format(self, record):
        colour = self.LEVEL_COLOURS.get(record.levelno, Colours.RESET)
        record.levelname = f"{colour}{record.levelname:<8}{Colours.RESET}"
        record.name      = f"{Colours.GREY}{record.name}{Colours.RESET}"
        return super().format(record)


def setup_logger(name: str = "trading_firm") -> logging.Logger:
    from config.settings import LOG_LEVEL, LOG_FILE

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if logger.handlers:
        return logger

    # Console handler (coloured)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColouredFormatter(
        fmt="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File handler (rotating, 5MB × 3 files)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_242_880, backupCount=3)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"trading_firm.{module_name}")
