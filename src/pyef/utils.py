import logging
from rich.logging import RichHandler
from ._config import get_option

__author__ = "Bhav Sardana"


def get_logger(
    name: str,
) -> logging.Logger:
    level = get_option("logging_level")
    logger = logging.getLogger(name)
    shell_handler = RichHandler()
    logger.setLevel(level)
    shell_handler.setLevel(level)
    fmt_shell = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    shell_formatter = logging.Formatter(fmt_shell)
    shell_handler.setFormatter(shell_formatter)
    logger.addHandler(shell_handler)
    return logger
