# Object Detectors Evaluation

## Installation

### Python dependencies

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and run scripts.

To install the dependencies, run the following command:

```bash
uv sync
```

If you want to install without development dependencies, run:

```bash
uv sync --no-dev
```

If you want to install the package with development dependencies, run:

```bash
uv sync --group dev
```

Or in general for any specific group of dependencies defined in `pyproject.toml`:

```bash
uv sync --group <GROUP_NAME>
```

To add a package to the project, use:

```bash
uv add <PACKAGE_NAME>
```

If you want to add a package with development dependencies, use:

```bash
uv add --group dev <PACKAGE_NAME>
```

If you want bump project version, use:

```bash
uv bump <VERSION>
```

Or specify which part of the version to bump:

```bash
uv version --bump <SEMVER_PART> # e.g. major, minor, patch
```

## Tests

Tests live in the `tests` directory.

Install development dependencies first:

```bash
uv sync --group dev
```

Run all tests:

```bash
uv run pytest tests
```

Run only specific test:

```bash
uv run pytest tests/test_<some_file>.py
```

Show `print()` output during test execution:

```bash
uv run pytest tests -s
```

Show logs in the terminal:

```bash
uv run pytest tests -o log_cli=true --log-cli-level=INFO
```

## Pre-commit

This project uses pre-commit to ensure that the code is formatted correctly.

To install pre-commit, run the following command:

```bash
pre-commit install
```

or

```bash
uv run pre-commit install
```

To run pre-commit on all files, run the following command:

```bash
pre-commit run --all-files
```

or

```bash
uv run pre-commit run --all-files
```
