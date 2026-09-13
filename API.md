# Public API

`survey-plots` provides three functions for producing consistently styled survey
diagnostic plots. The canonical implementation is
[`src/plots.py`](src/plots.py).

## Deployment and imports

The module may be installed as the `survey-plots` project or copied verbatim into
a downstream repository. Vendored copies should be taken from an explicitly
selected Git revision and must not be edited downstream.

When the project is installed directly, import from the top-level `plots` module:

```python
from plots import (
    OutlierMethod,
    OutlierTail,
    plot_nobs_by_id,
    plot_nobs_by_wave,
    plot_stats_by_wave,
)
```

When the file is vendored inside another package, adjust only the import path. For
example, a copy at `src/SCE/plots.py` is imported from `SCE.plots`.

Only names listed in `plots.__all__` are supported as public API. Underscore-prefixed
helpers, formatters, type aliases, and style dictionaries are implementation
details.

## Data contract

- `df_data` must already contain all columns requested by the call.
- `variables` must be a nonempty ordered sequence of numeric columns.
- Numeric columns may use NumPy or nullable pandas numeric dtypes and may contain
  missing values.
- `wave_column` identifies the groups used for wave-level calculations.
- `id_column` identifies stable survey cases when distinct-case counts are needed.
- `x_column`, when supplied, provides the plotted coordinate associated with each
  wave, such as an interview date.
- `weight_column`, when supplied, contains survey weights. Missing, nonfinite, zero,
  and negative weights are excluded together with their corresponding values.
- `variable_labels` maps column names to panel annotations.
- `value_labels` maps column names to mappings from numeric codes to labels for
  coded variables.

The calling project is responsible for preparing and validating its survey data.
The plotting functions do not provide alternate handling for absent columns,
duplicate identifiers, or arbitrary nonnumeric variables.

## Example data

```python
from pathlib import Path

import pandas as pd

from plots import OutlierMethod, OutlierTail
from plots import plot_nobs_by_id, plot_nobs_by_wave, plot_stats_by_wave


df_data = pd.DataFrame(
    {
        "case_id": [1, 1, 2, 2],
        "wave": [1, 2, 1, 2],
        "date": pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]
        ),
        "income": [40_000.0, 42_000.0, 55_000.0, 58_000.0],
        "weight": [1.0, 1.1, 0.9, 1.0],
    }
)
output_dir = Path("graphs/diagnostics")
```

## Observation counts by sample unit

`plot_nobs_by_id` plots the distribution of nonmissing observations per stable
sample unit.

```python
plot_nobs_by_id(
    df_data,
    ["income"],
    output_dir / "nobs-by-id.pdf",
    id_column="case_id",
    suptitle="Income observations per respondent",
    variable_labels={"income": "Household income"},
)
```

The function counts nonmissing rows for each variable within each sample unit and
plots a histogram of those counts.

## Observation counts by wave

`plot_nobs_by_wave` plots nonmissing observations for each survey wave.

```python
plot_nobs_by_wave(
    df_data,
    ["income"],
    output_dir / "nobs-by-wave.pdf",
    wave_column="wave",
    id_column="case_id",
    x_column="date",
    suptitle="Income sample size by wave",
    variable_labels={"income": "Household income"},
)
```

Without `id_column`, the function counts nonmissing rows. With `id_column`, it
counts distinct nonmissing cases. The wave itself is plotted on the horizontal
axis unless `x_column` is supplied.

## Statistics by wave

`plot_stats_by_wave` plots the mean, median, and interquartile range for each wave.
Statistics are unweighted unless `weight_column` is supplied.

```python
plot_stats_by_wave(
    df_data,
    ["income"],
    output_dir / "stats-by-wave.pdf",
    wave_column="wave",
    x_column="date",
    weight_column="weight",
    suptitle="Household income by wave",
    variable_labels={"income": "Household income"},
    compact_variables={"income"},
    outlier_method=OutlierMethod.QUANTILE,
    outlier_tails={"income": OutlierTail.UPPER},
)
```

`compact_variables` selects axes that use compact magnitude suffixes such as `k`,
`m`, `bn`, and `tr`.

### Mean trimming

`OutlierMethod` provides three mean-trimming methods:

- `OutlierMethod.NONE` retains every finite observation.
- `OutlierMethod.QUANTILE` excludes observations beyond the configured tail
  quantiles.
- `OutlierMethod.IQR` excludes observations beyond interquartile-range fences.

Trimming applies only to the mean. The median and interquartile range always use
the full valid sample.

A single `OutlierMethod` applies to every variable. A mapping applies methods by
variable, with missing mapping entries using `OutlierMethod.NONE`. This permits,
for example, IQR trimming for most variables and quantile trimming for variables
whose within-wave IQR is zero:

```python
outlier_method={
    "income": OutlierMethod.IQR,
    "numerical_response": OutlierMethod.QUANTILE,
}
```

`outlier_tail_fraction` applies to every variable configured with
`OutlierMethod.QUANTILE`; it has no effect on variables using another method.
Likewise, `outlier_iqr_factor` applies only to variables configured with
`OutlierMethod.IQR`.

`OutlierTail` controls which distribution tails are eligible:

- `OutlierTail.AUTO` infers a rule for each variable.
- `OutlierTail.NONE` disables trimming.
- `OutlierTail.UPPER` uses only the upper tail.
- `OutlierTail.LOWER` uses only the lower tail.
- `OutlierTail.TWO_SIDED` uses both tails.

A single `OutlierTail` applies to every variable. A mapping applies rules by
variable, with missing mapping entries using `OutlierTail.AUTO`.

Automatic inference disables trimming for labeled, categorical, Boolean, integer,
and indicator variables. It also disables trimming for variables bounded between
zero and one. Otherwise, nonnegative variables use the upper tail, nonpositive
variables use the lower tail, and variables spanning zero use both tails.

## Output behavior

Each function:

- Creates the output directory when necessary.
- Chooses the output format from the file extension.
- Replaces an existing file at the requested path.
- Closes the Matplotlib figure after writing it.
- Returns the written `Path`.

## Styling

The canonical module defines a fixed visual style through module-level style
dictionaries. These dictionaries are not public configuration API. Change styles
only in the canonical `survey-plots` repository, then deploy a selected canonical
revision to consumers.
