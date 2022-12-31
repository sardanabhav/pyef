"""logger
this module should not exist
"""

import logging

from rich.logging import RichHandler

from pyef._config import get_option


def get_logger(
    name: str,
) -> logging.Logger:
    """TODO add summary.

    Args:
        name (str): _description_

    Returns:
        logging.Logger: _description_
    """
    level = get_option("logging_level")
    logger = logging.getLogger(name)
    shell_handler = RichHandler()
    logger.setLevel(level)
    shell_handler.setLevel(level)
    fmt_shell = "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    shell_formatter = logging.Formatter(fmt_shell)
    shell_handler.setFormatter(shell_formatter)
    logger.addHandler(shell_handler)
    return logger
