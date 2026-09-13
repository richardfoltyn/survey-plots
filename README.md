# survey-plots

A single-file plotting library for survey-data diagnostics. The canonical module
is [`src/plots.py`](src/plots.py); it can be installed as the `survey-plots`
project or copied verbatim into another project.

The library supports longitudinal and repeated cross-section surveys, optional
survey weights, distinct-case counts, datetime or numeric waves, and configurable
mean trimming. All visual settings are module-level constants near the top of
`plots.py`.

## Development

```bash
uv sync
uv run pytest -n auto
uv run ruff check src/plots.py tests
uv run ty check src/plots.py tests
```
