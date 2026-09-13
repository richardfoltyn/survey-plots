"""Create reusable survey diagnostic plots.

- Plot observation counts by sample unit and survey wave.
- Plot weighted or unweighted wave-level descriptive statistics.
- Apply consistent grid, annotation, axis, and outlier-trimming behavior.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
import logging
from pathlib import Path
from textwrap import fill
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter
import numpy as np
import numpy.typing as npt
import pandas as pd

FIGURE_WIDTH = 16.2
ROW_HEIGHT = 2.835
NCOLS = 5
FONT_FAMILY = "serif"
FIGURE_TITLE_SIZE = 14
FIGURE_TITLE_WEIGHT = "bold"
FIGURE_TITLE_STYLE = "italic"
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
VARIABLE_LABEL_SIZE = 11
VARIABLE_LABEL_STYLE = "italic"
VARIABLE_LABEL_X = 0.02
VARIABLE_LABEL_Y = 0.98
VARIABLE_LABEL_WRAP_WIDTH = 38
MAX_X_TICKS = 5
MAX_Y_TICKS = 5
X_MARGIN = 0.0
Y_MARGIN = 0.05
TOP_Y_MARGIN = 0.05
INDICATOR_LOWER_MARGIN = 0.05
INDICATOR_UPPER_MARGIN = 0.05
CATEGORICAL_MARGIN = 0.25
GRID_COLOR = "grey"
GRID_LINESTYLE = ":"
COUNT_COLOR = "steelblue"
COUNT_LINE_WIDTH = 1.0
HISTOGRAM_EDGE_COLOR = "white"
HISTOGRAM_LINE_WIDTH = 0.4
HISTOGRAM_RELATIVE_WIDTH = 0.9
MEDIAN_COLOR = "steelblue"
MEAN_COLOR = "black"
MEDIAN_LINE_WIDTH = 1.0
MEAN_LINE_WIDTH = 0.75
MEDIAN_ALPHA = 0.8
MEAN_ALPHA = 0.7
MEDIAN_ZORDER = 50
MEAN_ZORDER = 100
STATISTIC_MARKER: str | None = None
MARKER_SIZE = 4.0
IQR_COLOR = "steelblue"
IQR_ALPHA = 0.25
IQR_LINE_WIDTH = 0.0
LEGEND_LOCATION = "upper left"
LEGEND_FRAME = False
DATE_MIN_TICKS = 3
DATE_MAX_TICKS = 5
OUTLIER_TAIL_FRACTION = 0.01
OUTLIER_IQR_FACTOR = 100.0


type PanelPlotter = Callable[[Axes, str], None]
type OutlierMethod = Literal["none", "quantile", "iqr"]
type OutlierTails = Literal["auto", "none", "upper", "lower", "two-sided"]
type OutlierTailConfig = OutlierTails | Mapping[str, OutlierTails]
type VariableLabels = Mapping[str, str]
type ValueLabelCode = int | float
type ValueLabels = Mapping[str, Mapping[ValueLabelCode, str]]


class SuffixFormatter(FuncFormatter):
    """Format tick values using compact suffixes (``k``, ``m``, ``bn``, ``tr``)."""

    _SCALES = (
        (1.0e12, "tr"),
        (1.0e9, "bn"),
        (1.0e6, "m"),
        (1.0e3, "k"),
    )

    def __init__(self, default: str | None = None) -> None:
        """Create a formatter that shortens large magnitudes.

        Parameters
        ----------
        default
            Format specifier used for values without a suffix, such as ``.2f``.
            If omitted, Matplotlib's default formatting is used.
        """
        self.default = default
        super().__init__(self._format_value)

    def _format_value(self, value: float, _position: int) -> str:
        """Format one tick value."""
        scale, suffix = next(
            ((scale, suffix) for scale, suffix in self._SCALES if abs(value) >= scale),
            (1.0, ""),
        )
        scaled = value / scale
        if scaled.is_integer():
            fmt = ".0f"
        elif not suffix and self.default is not None:
            fmt = self.default
        else:
            fmt = ""

        sign = "$-$" if scaled < 0.0 else ""
        return f"{sign}{abs(scaled):{fmt}}{suffix}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(default={self.default!r})"


def _as_float_array(values: pd.Series) -> npt.NDArray[np.float64]:
    """Convert a Series to a float array, representing missing values by NaN."""
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _variable_annotation(variable: str, labels: VariableLabels | None) -> str:
    """Return an attached variable label or the variable name."""
    if labels is None:
        return variable
    return labels.get(variable, variable)


def _is_indicator(values: pd.Series) -> bool:
    """Return whether the finite observations contain only zero and one."""
    observed = _as_float_array(values)
    observed = observed[np.isfinite(observed)]
    return observed.size > 0 and bool(np.isin(observed, (0.0, 1.0)).all())


def _valid_weighted_values(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return finite values with corresponding finite, positive weights."""
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    return value_array[valid], weight_array[valid]


def _unweighted_moments(
    values: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Calculate an unweighted mean, median, and interquartile endpoints."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    q1, median, q3 = np.quantile(valid, [0.25, 0.5, 0.75])
    return float(valid.mean()), float(median), float(q1), float(q3)


def _weighted_moments(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Calculate a weighted mean, median, and interquartile endpoints."""
    valid_values, valid_weights = _valid_weighted_values(values, weights)
    if valid_values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    q1, median, q3 = np.quantile(
        valid_values,
        [0.25, 0.5, 0.75],
        weights=valid_weights,
        method="inverted_cdf",
    )
    mean = np.average(valid_values, weights=valid_weights)
    return float(mean), float(median), float(q1), float(q3)


def _auto_outlier_tails(
    values: pd.Series,
    value_labels: Mapping[ValueLabelCode, str] | None,
) -> OutlierTails:
    """Infer applicable tails from metadata, dtype, and observed support."""
    if (
        value_labels
        or isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_integer_dtype(values.dtype)
        or _is_indicator(values)
    ):
        return "none"

    observed = _as_float_array(values)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0 or (observed.min() >= 0.0 and observed.max() <= 1.0):
        return "none"
    if observed.min() >= 0.0:
        return "upper"
    if observed.max() <= 0.0:
        return "lower"
    return "two-sided"


def _resolve_outlier_tails(
    df_data: pd.DataFrame,
    variable: str,
    value_labels: ValueLabels | None,
    config: OutlierTailConfig,
) -> OutlierTails:
    """Resolve an explicit or inferred tail rule for one variable."""
    tails = config if isinstance(config, str) else config.get(variable, "auto")
    if tails == "none":
        return "none"
    if tails == "upper":
        return "upper"
    if tails == "lower":
        return "lower"
    if tails == "two-sided":
        return "two-sided"

    labels = None if value_labels is None else value_labels.get(variable)
    return _auto_outlier_tails(df_data[variable], labels)


def _tail_quantile(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    quantile: float,
) -> float:
    """Calculate one unweighted or weighted empirical quantile."""
    if weights is None:
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return float("nan")
        return float(np.quantile(valid, quantile, method="inverted_cdf"))

    valid_values, valid_weights = _valid_weighted_values(values, weights)
    if valid_values.size == 0:
        return float("nan")
    return float(
        np.quantile(
            valid_values,
            quantile,
            weights=valid_weights,
            method="inverted_cdf",
        )
    )


def _quantile_outlier_mask(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    tails: OutlierTails,
    fraction: float,
) -> npt.NDArray[np.bool_]:
    """Identify observations in the configured nonzero distribution tails."""
    valid = np.isfinite(values)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0.0)
    mask = np.zeros(values.shape, dtype=bool)

    if tails in {"upper", "two-sided"}:
        selected = valid & (values > 0.0)
        upper = _tail_quantile(
            values[selected],
            weights[selected] if weights is not None else None,
            1.0 - fraction,
        )
        mask |= valid & (values > upper)

    if tails in {"lower", "two-sided"}:
        selected = valid & (values < 0.0)
        lower = _tail_quantile(
            values[selected],
            weights[selected] if weights is not None else None,
            fraction,
        )
        mask |= valid & (values < lower)

    return mask


def _iqr_outlier_mask(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    tails: OutlierTails,
    factor: float,
) -> npt.NDArray[np.bool_]:
    """Identify observations outside the configured interquartile fences."""
    valid = np.isfinite(values)
    if weights is None:
        _, _, q1, q3 = _unweighted_moments(values)
    else:
        valid &= np.isfinite(weights) & (weights > 0.0)
        _, _, q1, q3 = _weighted_moments(values, weights)

    mask = np.zeros(values.shape, dtype=bool)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0.0:
        return mask

    if tails in {"upper", "two-sided"}:
        mask |= valid & (values > q3 + factor * iqr)
    if tails in {"lower", "two-sided"}:
        mask |= valid & (values < q1 - factor * iqr)
    return mask


def _mean_outlier_mask(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    *,
    method: OutlierMethod,
    tails: OutlierTails,
    tail_fraction: float,
    iqr_factor: float,
) -> npt.NDArray[np.bool_]:
    """Identify observations excluded from a trimmed mean."""
    if method == "none" or tails == "none":
        return np.zeros(values.shape, dtype=bool)
    if method == "quantile":
        return _quantile_outlier_mask(values, weights, tails, tail_fraction)
    return _iqr_outlier_mask(values, weights, tails, iqr_factor)


def _wave_coordinate(
    df_wave: pd.DataFrame,
    wave: object,
    x_column: str | None,
) -> object:
    """Return the plotted coordinate for one survey wave."""
    if x_column is None:
        return wave
    return df_wave[x_column].median()


def _nobs_by_wave(
    df_data: pd.DataFrame,
    variable: str,
    *,
    wave_column: str,
    id_column: str | None,
    x_column: str | None,
) -> pd.DataFrame:
    """Count nonmissing rows or distinct cases by survey wave."""
    rows: list[dict[str, object]] = []
    for wave, df_wave in df_data.groupby(
        wave_column,
        sort=True,
        observed=False,
    ):
        if id_column is None:
            nobs = int(df_wave[variable].count())
        else:
            selected = df_wave[variable].notna()
            nobs = int(df_wave.loc[selected, id_column].nunique())
        rows.append(
            {
                "x": _wave_coordinate(df_wave, wave, x_column),
                "nobs": nobs,
            }
        )

    return pd.DataFrame(rows, columns=("x", "nobs"))


def _stats_by_wave(
    df_data: pd.DataFrame,
    variable: str,
    *,
    wave_column: str,
    x_column: str | None,
    weight_column: str | None,
    value_labels: ValueLabels | None,
    outlier_method: OutlierMethod,
    outlier_tails: OutlierTailConfig,
    outlier_tail_fraction: float,
    outlier_iqr_factor: float,
) -> tuple[pd.DataFrame, int, OutlierTails]:
    """Calculate wave moments and optionally trim the mean."""
    rows: list[dict[str, object]] = []
    n_masked = 0
    tails = _resolve_outlier_tails(
        df_data,
        variable,
        value_labels,
        outlier_tails,
    )

    for wave, df_wave in df_data.groupby(
        wave_column,
        sort=True,
        observed=False,
    ):
        values = _as_float_array(df_wave[variable])
        weights = (
            _as_float_array(df_wave[weight_column])
            if weight_column is not None
            else None
        )
        if weights is None:
            mean, median, q1, q3 = _unweighted_moments(values)
        else:
            mean, median, q1, q3 = _weighted_moments(values, weights)

        mask = _mean_outlier_mask(
            values,
            weights,
            method=outlier_method,
            tails=tails,
            tail_fraction=outlier_tail_fraction,
            iqr_factor=outlier_iqr_factor,
        )
        if np.any(mask):
            n_masked += int(np.count_nonzero(mask))
            retained = values.copy()
            retained[mask] = np.nan
            if weights is None:
                mean, _, _, _ = _unweighted_moments(retained)
            else:
                mean, _, _, _ = _weighted_moments(retained, weights)

        rows.append(
            {
                "x": _wave_coordinate(df_wave, wave, x_column),
                "mean": mean,
                "median": median,
                "q1": q1,
                "q3": q3,
            }
        )

    df_result = pd.DataFrame(
        rows,
        columns=("x", "mean", "median", "q1", "q3"),
    )
    return df_result, n_masked, tails


def _style_x_axis(
    ax: Axes,
    values: pd.Series,
    ticks: Sequence[float] | None,
) -> None:
    """Format a datetime or numeric survey-wave axis."""
    ax.margins(x=X_MARGIN)
    if ticks is not None:
        ax.set_xticks(np.asarray(ticks, dtype=np.float64))
    elif pd.api.types.is_datetime64_any_dtype(values.dtype):
        locator = AutoDateLocator(minticks=DATE_MIN_TICKS, maxticks=DATE_MAX_TICKS)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    elif pd.api.types.is_numeric_dtype(values.dtype):
        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins=MAX_X_TICKS - 1,
                integer=pd.api.types.is_integer_dtype(values.dtype),
                min_n_ticks=1,
            )
        )
    ax.tick_params(axis="x", labelrotation=0, labelsize=TICK_LABEL_SIZE)


def _add_top_clearance(ax: Axes) -> None:
    """Reserve clearance between plotted values and the panel annotation."""
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + TOP_Y_MARGIN * (top - bottom))


def _style_count_axis(ax: Axes) -> None:
    """Format a nonnegative integer count axis."""
    ax.margins(y=Y_MARGIN)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=MAX_Y_TICKS - 1,
            integer=True,
            min_n_ticks=1,
        )
    )
    ax.yaxis.set_major_formatter(SuffixFormatter())
    ax.tick_params(axis="y", labelrotation=90, labelsize=TICK_LABEL_SIZE)
    _add_top_clearance(ax)


def _style_stat_axis(
    ax: Axes,
    values: pd.Series,
    *,
    value_labels: Mapping[ValueLabelCode, str] | None,
    compact: bool,
) -> None:
    """Format an axis displaying descriptive statistics."""
    indicator = _is_indicator(values)
    ax.margins(y=Y_MARGIN)
    ax.tick_params(axis="y", labelrotation=90, labelsize=TICK_LABEL_SIZE)

    if indicator:
        ax.set_ylim(-INDICATOR_LOWER_MARGIN, 1.0 + INDICATOR_UPPER_MARGIN)
        ax.set_yticks(np.linspace(0.0, 1.0, MAX_Y_TICKS))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    elif value_labels:
        codes = np.array(sorted(value_labels), dtype=np.float64)
        ax.set_ylim(codes[0] - CATEGORICAL_MARGIN, codes[-1] + CATEGORICAL_MARGIN)
        ax.set_yticks(codes)
    elif compact:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS - 1, min_n_ticks=1))
        ax.yaxis.set_major_formatter(SuffixFormatter())
    elif pd.api.types.is_integer_dtype(values.dtype):
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=MAX_Y_TICKS - 1,
                integer=True,
                min_n_ticks=1,
            )
        )
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS - 1, min_n_ticks=1))

    _add_top_clearance(ax)


def _annotate_panel(ax: Axes, annotation: str) -> None:
    """Place a wrapped variable annotation in the upper-left panel corner."""
    wrapped = "\n".join(
        fill(line, width=VARIABLE_LABEL_WRAP_WIDTH) for line in annotation.splitlines()
    )
    text = ax.annotate(
        wrapped,
        xy=(VARIABLE_LABEL_X, VARIABLE_LABEL_Y),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=VARIABLE_LABEL_SIZE,
        fontstyle=VARIABLE_LABEL_STYLE,
    )
    text.set_in_layout(False)


def _draw_descriptive_legend(ax: Axes) -> None:
    """Draw the descriptive-statistics legend in a blank panel."""
    handles = (
        Line2D(
            [],
            [],
            color=MEDIAN_COLOR,
            linewidth=MEDIAN_LINE_WIDTH,
            marker=STATISTIC_MARKER,
            markersize=MARKER_SIZE,
        ),
        Line2D(
            [],
            [],
            color=MEAN_COLOR,
            linewidth=MEAN_LINE_WIDTH,
            marker=STATISTIC_MARKER,
            markersize=MARKER_SIZE,
        ),
        Patch(facecolor=IQR_COLOR, alpha=IQR_ALPHA, edgecolor="none"),
    )
    ax.set_axis_off()
    ax.legend(
        handles,
        ("Median", "Mean", "IQR"),
        loc=LEGEND_LOCATION,
        frameon=LEGEND_FRAME,
        fontsize=TICK_LABEL_SIZE,
    )


def _write_grid(
    variables: Sequence[str],
    output_path: Path | str,
    plot_panel: PanelPlotter,
    *,
    suptitle: str,
    descriptive_legend: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Write one diagnostic grid and close its figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = int(descriptive_legend)
    nrows = (len(variables) + first + NCOLS - 1) // NCOLS

    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig, axes = plt.subplots(
            nrows,
            NCOLS,
            figsize=(FIGURE_WIDTH, ROW_HEIGHT * nrows),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        try:
            if descriptive_legend:
                _draw_descriptive_legend(axes[0, 0])

            plot_axes = tuple(axes.flat)[first:]
            for ax, variable in zip(plot_axes, variables, strict=False):
                plot_panel(ax, variable)

            for ax in plot_axes[len(variables) :]:
                ax.set_visible(False)

            if xlabel is not None:
                fig.supxlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
            if ylabel is not None:
                fig.supylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
            fig.suptitle(
                suptitle,
                fontsize=FIGURE_TITLE_SIZE,
                fontweight=FIGURE_TITLE_WEIGHT,
                fontstyle=FIGURE_TITLE_STYLE,
            )
            fig.savefig(path)
        finally:
            plt.close(fig)

    logging.getLogger("PLOTS").info("Wrote diagnostic plot: %s", path)
    return path


def plot_nobs_by_id(
    df_data: pd.DataFrame,
    variables: Sequence[str],
    output_path: Path | str,
    *,
    id_column: str,
    suptitle: str,
    variable_labels: VariableLabels | None = None,
    xlabel: str = "Nonmissing observations per sample unit",
    ylabel: str = "Sample units",
) -> Path:
    """Plot the distribution of nonmissing observations per sample unit.

    Parameters
    ----------
    df_data
        Data containing a stable panel identifier and requested variables.
    variables
        Ordered variables assigned to plot panels.
    output_path
        Destination figure path.
    id_column
        Column containing the stable panel identifier.
    suptitle
        Figure title.
    variable_labels
        Optional labels keyed by variable name.
    xlabel
        Shared horizontal-axis label.
    ylabel
        Shared vertical-axis label.

    Returns
    -------
    Written figure path.
    """
    df_nobs = df_data.groupby(id_column, sort=False)[list(variables)].count()
    nmax = int(df_data.groupby(id_column, sort=False).size().max())
    bins = np.arange(-0.5, nmax + 1.5)

    def plot_panel(ax: Axes, variable: str) -> None:
        """Draw one sample-unit observation-count histogram."""
        ax.hist(
            df_nobs[variable].to_numpy(),
            bins=bins.tolist(),
            color=COUNT_COLOR,
            linewidth=HISTOGRAM_LINE_WIDTH,
            rwidth=HISTOGRAM_RELATIVE_WIDTH,
            edgecolor=HISTOGRAM_EDGE_COLOR,
        )
        ax.set_xticks(np.arange(nmax + 1))
        ax.set_xlim(-0.75, nmax + 0.25)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        _style_count_axis(ax)
        _annotate_panel(ax, _variable_annotation(variable, variable_labels))

    return _write_grid(
        variables,
        output_path,
        plot_panel,
        suptitle=suptitle,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def plot_nobs_by_wave(
    df_data: pd.DataFrame,
    variables: Sequence[str],
    output_path: Path | str,
    *,
    wave_column: str,
    suptitle: str,
    id_column: str | None = None,
    x_column: str | None = None,
    x_ticks: Sequence[float] | None = None,
    variable_labels: VariableLabels | None = None,
    xlabel: str = "Wave",
    ylabel: str = "Number of observations",
) -> Path:
    """Plot nonmissing row or distinct-case counts by survey wave.

    Parameters
    ----------
    df_data
        Data containing the wave, optional case ID, and requested variables.
    variables
        Ordered variables assigned to plot panels.
    output_path
        Destination figure path.
    wave_column
        Column used to group survey waves.
    suptitle
        Figure title.
    id_column
        Optional case identifier. When supplied, distinct nonmissing cases are
        counted instead of rows.
    x_column
        Optional plotted wave coordinate, such as a survey date. The wave
        column itself is used when omitted.
    x_ticks
        Optional explicit horizontal tick positions.
    variable_labels
        Optional labels keyed by variable name.
    xlabel
        Shared horizontal-axis label.
    ylabel
        Shared vertical-axis label.

    Returns
    -------
    Written figure path.
    """

    def plot_panel(ax: Axes, variable: str) -> None:
        """Draw nonmissing counts for one variable."""
        df_stats = _nobs_by_wave(
            df_data,
            variable,
            wave_column=wave_column,
            id_column=id_column,
            x_column=x_column,
        )
        ax.plot(
            df_stats["x"],
            df_stats["nobs"],
            color=COUNT_COLOR,
            linewidth=COUNT_LINE_WIDTH,
            marker=STATISTIC_MARKER,
            markersize=MARKER_SIZE,
        )
        ax.grid(color=GRID_COLOR, linestyle=GRID_LINESTYLE)
        _style_x_axis(ax, df_stats["x"], x_ticks)
        _style_count_axis(ax)
        _annotate_panel(ax, _variable_annotation(variable, variable_labels))

    return _write_grid(
        variables,
        output_path,
        plot_panel,
        suptitle=suptitle,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def plot_stats_by_wave(
    df_data: pd.DataFrame,
    variables: Sequence[str],
    output_path: Path | str,
    *,
    wave_column: str,
    suptitle: str,
    x_column: str | None = None,
    x_ticks: Sequence[float] | None = None,
    weight_column: str | None = None,
    variable_labels: VariableLabels | None = None,
    value_labels: ValueLabels | None = None,
    compact_variables: Collection[str] = (),
    outlier_method: OutlierMethod = "none",
    outlier_tails: OutlierTailConfig = "auto",
    outlier_tail_fraction: float = OUTLIER_TAIL_FRACTION,
    outlier_iqr_factor: float = OUTLIER_IQR_FACTOR,
    xlabel: str = "Wave",
    ylabel: str | None = None,
) -> Path:
    """Plot means, medians, and interquartile ranges by survey wave.

    Parameters
    ----------
    df_data
        Data containing waves, requested variables, and an optional weight.
    variables
        Ordered variables assigned to plot panels.
    output_path
        Destination figure path.
    wave_column
        Column used to group survey waves.
    suptitle
        Figure title.
    x_column
        Optional plotted wave coordinate, such as a survey date. The wave
        column itself is used when omitted.
    x_ticks
        Optional explicit horizontal tick positions.
    weight_column
        Optional positive survey-weight column used for every statistic.
    variable_labels
        Optional labels keyed by variable name.
    value_labels
        Optional categorical value labels keyed by variable and code.
    compact_variables
        Variables whose axes use compact magnitude suffixes.
    outlier_method
        Mean-trimming method. Median and interquartile ranges always use the
        full valid sample.
    outlier_tails
        Global tail rule or per-variable tail rules. Missing mapping entries
        use automatic inference.
    outlier_tail_fraction
        Fraction removed from each applicable nonzero tail by the quantile
        method.
    outlier_iqr_factor
        Interquartile-range multiplier used by the IQR method.
    xlabel
        Shared horizontal-axis label.
    ylabel
        Optional shared vertical-axis label.

    Returns
    -------
    Written figure path.
    """
    logger = logging.getLogger("PLOTS")

    def plot_panel(ax: Axes, variable: str) -> None:
        """Draw descriptive statistics for one variable."""
        df_stats, n_masked, tails = _stats_by_wave(
            df_data,
            variable,
            wave_column=wave_column,
            x_column=x_column,
            weight_column=weight_column,
            value_labels=value_labels,
            outlier_method=outlier_method,
            outlier_tails=outlier_tails,
            outlier_tail_fraction=outlier_tail_fraction,
            outlier_iqr_factor=outlier_iqr_factor,
        )
        if outlier_method != "none":
            logger.info(
                "Mean trimming for %s (%s, %s): %d rows masked",
                variable,
                outlier_method,
                tails,
                n_masked,
            )

        x = df_stats["x"]
        mean = df_stats["mean"].to_numpy(dtype=np.float64)
        median = df_stats["median"].to_numpy(dtype=np.float64)
        q1 = df_stats["q1"].to_numpy(dtype=np.float64)
        q3 = df_stats["q3"].to_numpy(dtype=np.float64)

        ax.fill_between(
            x,
            q1,
            q3,
            color=IQR_COLOR,
            alpha=IQR_ALPHA,
            linewidth=IQR_LINE_WIDTH,
        )
        ax.plot(
            x,
            median,
            color=MEDIAN_COLOR,
            linewidth=MEDIAN_LINE_WIDTH,
            marker=STATISTIC_MARKER,
            markersize=MARKER_SIZE,
            alpha=MEDIAN_ALPHA,
            zorder=MEDIAN_ZORDER,
        )
        ax.plot(
            x,
            mean,
            color=MEAN_COLOR,
            linewidth=MEAN_LINE_WIDTH,
            marker=STATISTIC_MARKER,
            markersize=MARKER_SIZE,
            alpha=MEAN_ALPHA,
            zorder=MEAN_ZORDER,
        )
        _style_x_axis(ax, x, x_ticks)
        labels = None if value_labels is None else value_labels.get(variable)
        _style_stat_axis(
            ax,
            df_data[variable],
            value_labels=labels,
            compact=variable in compact_variables,
        )
        _annotate_panel(ax, _variable_annotation(variable, variable_labels))

    return _write_grid(
        variables,
        output_path,
        plot_panel,
        suptitle=suptitle,
        descriptive_legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
    )
