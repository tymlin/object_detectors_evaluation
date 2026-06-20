import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from object_detectors_evaluation.loggers import logger


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    :param filepath: JSON filepath.
    :return: Loaded JSON object.
    """
    filepath = Path(filepath)
    require_filepath(filepath=filepath, description="JSON")
    logger.info(f"Loading JSON file from path: '{filepath}'")
    with filepath.open() as file:
        return json.load(file)


def load_jsonl(filepath: str | Path) -> list[Any]:
    """Load records from a JSON Lines file.

    :param filepath: JSON Lines filepath.
    :return: Loaded JSON-compatible records.
    """
    return list(iter_jsonl(filepath=filepath))


def iter_jsonl(filepath: str | Path) -> Iterator[Any]:
    """Iterate over records from a JSON Lines file.

    :param filepath: JSON Lines filepath.
    :return: Iterator over JSON-compatible records.
    """
    filepath = Path(filepath)
    require_filepath(filepath=filepath, description="JSON Lines")
    logger.info(f"Reading JSON Lines file from path: '{filepath}'")
    with filepath.open() as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def save_json(filepath: str | Path, data: object, indent: int = 4) -> None:
    """Save data as JSON.

    :param filepath: Output JSON filepath.
    :param data: JSON-serializable data.
    :param indent: JSON indentation.
    """
    filepath = Path(filepath)
    require_dirpath(dirpath=filepath.parent, description="parent")
    logger.info(f"Saving JSON file to path: '{filepath}'")
    with filepath.open("w") as file:
        json.dump(data, file, indent=indent)


def append_jsonl(filepath: str | Path, records: Iterable[object]) -> None:
    """Append records to a JSON Lines file.

    :param filepath: Output JSON Lines filepath.
    :param records: JSON-serializable records.
    """
    filepath = Path(filepath)
    require_dirpath(dirpath=filepath.parent, description="parent")
    logger.debug(f"Appending JSON Lines file at path: '{filepath}'")
    with filepath.open("a") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def save_text(filepath: str | Path, text: str) -> None:
    """Save text to a file.

    :param filepath: Output text filepath.
    :param text: Text content.
    """
    filepath = Path(filepath)
    require_dirpath(dirpath=filepath.parent, description="parent")
    logger.info(f"Saving text file to path: '{filepath}'")
    filepath.write_text(text)


def load_yaml(filepath: str | Path) -> Any:
    """Load YAML data from disk.

    :param filepath: YAML filepath.
    :return: Loaded YAML data.
    """
    filepath = Path(filepath)
    require_filepath(filepath=filepath, description="YAML")
    logger.info(f"Loading YAML file from path: '{filepath}'")
    with filepath.open() as file:
        return yaml.safe_load(file)


def save_yaml(filepath: str | Path, data: object, sort_keys: bool = False) -> None:
    """Save data as YAML.

    :param filepath: Output YAML filepath.
    :param data: YAML-serializable data.
    :param sort_keys: Whether YAML keys should be sorted.
    """
    filepath = Path(filepath)
    require_dirpath(dirpath=filepath.parent, description="parent")
    logger.info(f"Saving YAML file to path: '{filepath}'")
    with filepath.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=sort_keys)


def require_dirpath(dirpath: str | Path, description: str = "") -> Path:
    """Require an existing directory path.

    :param dirpath: Directory path.
    :param description: Optional path description used in error messages.
    :return: Directory path.
    """
    dirpath = Path(dirpath)
    if dirpath.is_dir():
        return dirpath

    description = f"{description} " if description else ""
    msg = f"Could not find {description}directory at path: '{dirpath}'"
    logger.error(msg)
    raise FileNotFoundError(msg)


def require_filepath(filepath: str | Path, description: str = "") -> Path:
    """Require an existing file path.

    :param filepath: File path.
    :param description: Optional path description used in error messages.
    :return: File path.
    """
    filepath = Path(filepath)
    if filepath.is_file():
        return filepath

    description = f"{description} " if description else ""
    msg = f"Could not find {description}file at path: '{filepath}'"
    logger.error(msg)
    raise FileNotFoundError(msg)
