# Project-Specific Instructions for AI Agents

- Follow the instructions in the global user-configuration instructions.
- This file contains additional project-specific instructions.

## Python

- This project uses `uv` to manage the local Python environment.
- Do not attempt to install additional packages in the local `venv`, unless explicitly asked to do so.
- `ruff` and `ty` for type checking are installed directly in the operating system. Run them using the `uv run` prefix (e.g., `uv run ty check` and `uv run ruff check`) to ensure they use the local `.venv` environment and resolve dependencies correctly. Do not attempt to install them in the local environment.

### Execution and unit tests

- Run unit tests only at the **end** of a task, if applicable.
- When applicable, run pytest targets in parallel with pytest-xdist, using
  `uv run pytest -n auto <test paths>`.
