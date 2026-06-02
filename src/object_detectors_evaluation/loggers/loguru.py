from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from ..consts import LOGGER_FILENAME, LOGGER_FORMAT, LOGGER_LEVEL, RUNS_DIRPATH

logger = _logger


def get_log_filepath(
    run_name: str | None = None,
    run_dirpath: str | Path | None = None,
    log_filename: str | None = LOGGER_FILENAME,
) -> Path | None:
    """Resolve the logfile path for a run.

    :param run_name: Name of the run directory inside ``RUNS_DIRPATH``.
    :param run_dirpath: Explicit run directory path.
    :param log_filename: Filename for the log sink. If ``None``, file logging is disabled.
    :return: Path to the log file for the selected run, or ``None`` if file logging is disabled.
    :raises ValueError: If both ``run_name`` and ``run_dirpath`` are provided.
    """
    if run_name is not None and run_dirpath is not None:
        raise ValueError("Provide either 'run_name' or 'run_dirpath', not both.")

    if log_filename is None:
        return None

    if run_dirpath is not None:
        run_dir = Path(run_dirpath)
    elif run_name is not None:
        run_dir = RUNS_DIRPATH / run_name
    else:
        return None

    return run_dir / log_filename


def configure_logger(
    run_name: str | None = None,
    run_dirpath: str | Path | None = None,
    log_filename: str | None = LOGGER_FILENAME,
    level: str = LOGGER_LEVEL,
    format_: str = LOGGER_FORMAT,
    add_console_sink: bool = True,
) -> Path | None:
    """Configure the shared Loguru logger for console and optional per-run file logging.

    :param run_name: Name of the run directory inside ``RUNS_DIRPATH``.
    :param run_dirpath: Explicit run directory path.
    :param log_filename: Filename for the log sink. If ``None``, file logging is disabled.
    :param level: Logging level applied to configured sinks.
    :param format_: Loguru format string.
    :param add_console_sink: Whether to keep a stdout sink.
    :return: The resolved log filepath, or ``None`` if no file sink is configured.
    """
    log_filepath = get_log_filepath(
        run_name=run_name,
        run_dirpath=run_dirpath,
        log_filename=log_filename,
    )

    logger.remove()

    if add_console_sink:
        logger.add(
            sys.stdout,
            colorize=True,
            level=level,
            format=format_,
        )

    if log_filepath is not None:
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_filepath,
            colorize=False,
            level=level,
            format=format_,
        )

    return log_filepath


configure_logger()


if __name__ == "__main__":
    logger.info("Logger configured")
    logger.debug("Debug example for {}", "video metadata")
    logger.warning("Warning example for {}", "missing optional field")
    logger.error("Error example for {}", "failed request")
