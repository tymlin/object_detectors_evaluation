from pathlib import Path

import pytest

from object_detectors_evaluation.consts import RUNS_DIRPATH
from object_detectors_evaluation.loggers import configure_logger, get_log_filepath, logger


@pytest.fixture(autouse=True)
def restore_logger() -> None:
    configure_logger(log_filename=None, add_console_sink=False)
    yield
    configure_logger(log_filename=None, add_console_sink=False)


def test_get_log_filepath_resolves_run_name() -> None:
    log_filepath = get_log_filepath(run_name="smoke-run", log_filename="pipeline.log")

    assert log_filepath == RUNS_DIRPATH / "smoke-run" / "pipeline.log"


def test_get_log_filepath_resolves_explicit_run_dirpath(tmp_path: Path) -> None:
    run_dirpath = tmp_path / "run-001"

    log_filepath = get_log_filepath(run_dirpath=run_dirpath, log_filename="pipeline.log")

    assert log_filepath == run_dirpath / "pipeline.log"


def test_get_log_filepath_returns_none_when_file_logging_disabled(tmp_path: Path) -> None:
    log_filepath = get_log_filepath(run_dirpath=tmp_path, log_filename=None)

    assert log_filepath is None


def test_get_log_filepath_rejects_run_name_and_run_dirpath(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Provide either 'run_name' or 'run_dirpath'"):
        get_log_filepath(run_name="smoke-run", run_dirpath=tmp_path, log_filename="pipeline.log")


def test_configure_logger_writes_to_file(tmp_path: Path) -> None:
    run_dirpath = tmp_path / "run-001"

    log_filepath = configure_logger(
        run_dirpath=run_dirpath,
        log_filename="pipeline.log",
        level="INFO",
        format_="{level}|{message}",
        add_console_sink=False,
    )
    logger.info("tracker {}", "ready")
    logger.complete()

    assert log_filepath == run_dirpath / "pipeline.log"
    assert log_filepath.exists()
    assert log_filepath.read_text() == "INFO|tracker ready\n"


def test_logger_prints_standard_examples(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logger(
        log_filename=None,
        level="DEBUG",
        format_="{message}",
        add_console_sink=True,
    )

    logger.info("Logger configured")
    logger.debug("Debug example for {}", "video metadata")
    logger.warning("Warning example for {}", "missing optional field")
    logger.error("Error example for {}", "failed request")
    logger.complete()

    assert capsys.readouterr().out.splitlines() == [
        "Logger configured",
        "Debug example for video metadata",
        "Warning example for missing optional field",
        "Error example for failed request",
    ]
