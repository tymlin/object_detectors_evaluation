## Build, Test, and Development Commands

- `uv sync` installs project dependencies.
- `uv sync --group dev` installs dev tooling such as pre-commit and pytest.
- `uv run pre-commit install` installs commit hooks locally.
- `uv build` builds a distributable package using `uv_build`.
- Codex must not run pre-commit unless explicitly requested. It is run
  automatically on commits.
- Codex must not run tests or Python scripts for code-correctness checks unless
  explicitly requested. These are run in the remote environment.
- Codex must not install packages or create Python environments. Only report
  missing dependencies and what is required.

## General Coding Style & Naming Conventions

- Python 3.13.
- Type hints are encouraged.
- Formatter/linter: Ruff via pre-commit, with line length 120.
- Indentation: 4 spaces.
- UTF-8 files should end with a newline.
- Naming: modules/functions `snake_case`; classes `CamelCase`; constants
  `UPPER_SNAKE_CASE`.
- Always use explicit imports. Do not use `*` imports.
- Keep imports at the top of the file unless a local import is needed to avoid a
  circular dependency or optional dependency import cost.
- Relative imports within the same package are allowed.
- For packages, expose public APIs through `__init__.py` files. In other
  packages, prefer importing from the package API rather than deep module paths.
- Avoid circular imports by refactoring code into separate modules or using
  carefully scoped local imports.
- Use f-strings and `pathlib.Path` for paths.
- Do not use old Python typing aliases such as `List` or `Dict`; use `list`,
  `dict`, and other built-in generic types.
- Use Sphinx-style docstrings with `:param`, `:return:`, and related fields when
  docstrings are useful.
- While creating Pydantic models, use `Field(...)` with descriptions. If fields
  require constraints such as minimum or maximum values, use the appropriate
  `Field` arguments.

Use these names when operating on paths:

```python
from pathlib import Path

dirpath = Path("/some/directory")
dirname = dirpath.name
filepath = dirpath / "file.txt"
filename = filepath.name
filestem = filepath.stem
filesuffix = filepath.suffix
```

## Project-Specific Coding Conventions

- Expand shared utility modules under `src/object_detectors_evaluation/` as needed rather
  than introducing unrelated helper locations.
- Use the existing Loguru logger object named `logger` for logging instead of
  prints when logging infrastructure is available.
- Do not use emojis in log messages.
- If logging a path, wrap it in single quotes:

```python
logger.info(f"Loading from path: '{filepath}'")
```

- If logging an object name or identifier, wrap it in backticks:

```python
logger.info(f"Starting module `{module_name}`")
```

- If raising an error, first create a `msg` variable, log the message, and then
  raise the error:

```python
msg = "This is an error"
logger.error(msg)
raise ValueError(msg)
```

## Testing Guidelines

- For new tests, prefer `pytest`-style files named `test_*.py`.
- Keep tests focused on deterministic model logic, task behavior, and schema
  compatibility.
- No coverage threshold is enforced yet.
- Reference tests that require AGPL or heavy optional dependencies should be
  opt-in and not required for normal CI.
- Codex should not run tests unless explicitly requested by the user.

## Commit & Pull Request Guidelines

- Match existing history style where possible: short, lower-case,
  action-oriented subjects.
- Keep commits scoped to one logical change.
- PRs should include purpose, key code paths changed, and local commands run.

## PR message template

If asked to write a PR message, use the following template:

```text
This pull request adds [short summary of the main change].

It introduces [brief description of the primary implementation], updates [secondary area if applicable], and includes [supporting changes, cleanup, or follow-up work]. It also improves [expected outcome or practical impact].

**[Primary change area]**

- Added [main feature, workflow, or capability].
- Updated [related module, component, or behavior].
- Improved [performance, reliability, maintainability, or usability].
- Added support for [new input, mode, config, or edge case].

**[Secondary change area]**

- Refactored [shared logic, structure, or API surface].
- Standardized [naming, flow, output format, or configuration].
- Exposed [new classes, functions, scripts, or exports] where needed.
- Aligned [related modules or scripts] with the new behavior.

... [3rd, 4th, ..., other changes]

**[Supporting updates]**

- Added logging, validation, or error handling around [relevant workflow].
- Updated example scripts, utilities, or documentation for the new flow.
- Added result/reporting/visualization support for [new metrics or outputs].
- Included minor cleanup and consistency updates in related code.
```

## Security & Configuration Tips

- Never commit secrets, raw credentials, or local cache artifacts.
- Use `.env` for local secrets where applicable, and do not commit it.
- Verify CUDA/device runtime behavior when using GPU execution.

## Rules for Codex

- Do not install packages or create Python environments.
- Only report missing dependencies and what is required.
- Do not run pre-commit unless explicitly requested.
- Do not run tests or Python scripts for checking code correctness unless
  explicitly requested.
- The user handles installing packages, managing virtualenvs, and system setup.
- Codex may edit source code and test files, but must not modify configs or
  system/global settings without asking.
- Before shell commands, read this file and every referenced instruction file.

## Getting Help

If unsure about something:

- Read existing code and find similar patterns in the codebase.
- Check types and analyze data structures and Pydantic models.
- Ask for clarification when assumptions would be risky.
