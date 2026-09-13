# survey-plots

A single-file plotting library for survey-data diagnostics. The canonical module
is [`src/plots.py`](src/plots.py); it can be installed as the `survey-plots`
project or copied verbatim into another project.

The library supports longitudinal and repeated cross-section surveys, optional
survey weights, distinct-case counts, datetime or numeric waves, and configurable
mean trimming. Rendering styles are fixed in the canonical module so vendored
copies produce consistent output.

See [`API.md`](API.md) for the supported public interface, data requirements, and
usage examples.

## Development

```bash
uv sync
uv run pytest -n auto
uv run ruff check src/plots.py tests
uv run ty check src/plots.py tests
```

## License

This project is licensed under the GNU Lesser General Public License, version 3
or later. See [`LICENSE.txt`](LICENSE.txt) for the full license text.
