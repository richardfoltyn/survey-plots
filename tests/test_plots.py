"""Tests for generic survey plotting and statistical helpers."""

from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plots


def test_nullable_series_convert_to_float_array() -> None:
    """Represent nullable integer and Boolean values as floats and NaN."""
    integer = pd.Series([1, pd.NA, 0], dtype="Int64")
    boolean = pd.Series([True, pd.NA, False], dtype="boolean")

    np.testing.assert_equal(plots._as_float_array(integer), [1.0, np.nan, 0.0])
    np.testing.assert_equal(plots._as_float_array(boolean), [1.0, np.nan, 0.0])


def test_indicator_detection_accepts_nullable_and_float_storage() -> None:
    """Recognize binary values independently of their numeric storage dtype."""
    nullable = pd.Series([0, 1, pd.NA], dtype="Int8")
    floating = pd.Series([0.0, 1.0, np.nan])
    nonbinary = pd.Series([0.0, 0.5, 1.0])

    assert plots._is_indicator(nullable)
    assert plots._is_indicator(floating)
    assert not plots._is_indicator(nonbinary)


def test_suffix_formatter_compacts_large_magnitudes() -> None:
    """Use compact suffixes while retaining an explicit negative sign."""
    formatter = plots.SuffixFormatter()

    assert formatter(0.0, 0) == "0"
    assert formatter(12.5, 0) == "12.5"
    assert formatter(1_500.0, 0) == "1.5k"
    assert formatter(2_000_000.0, 0) == "2m"
    assert formatter(-2_500_000_000.0, 0) == "$-$2.5bn"
    assert formatter(1_000_000_000_000.0, 0) == "1tr"
    assert plots.SuffixFormatter(default=".2f")(12.5, 0) == "12.50"


def test_isolated_point_mask_finds_only_points_without_valid_neighbors() -> None:
    """Find boundary and internal singletons while excluding valid runs."""
    cases = (
        ([], []),
        ([True], [True]),
        ([False], [False]),
        ([True, True], [False, False]),
        ([True, False], [True, False]),
        ([False, True], [False, True]),
        ([True, False, True], [True, False, True]),
    )

    for valid, expected in cases:
        mask = plots._isolated_point_mask(np.array(valid, dtype=bool))
        np.testing.assert_array_equal(mask, expected)


def test_weighted_moments_drop_invalid_weights_and_aligned_values() -> None:
    """Apply one joint validity rule to weighted means and quantiles."""
    values = np.array([0.0, 10.0, 100.0, np.nan, 500.0])
    weights = np.array([1.0, 3.0, 0.0, 4.0, np.nan])

    mean, median, q1, q3 = plots._weighted_moments(values, weights)

    assert mean == pytest.approx(7.5)
    assert median == 10.0
    assert q1 == 0.0
    assert q3 == 10.0


def test_wave_counts_can_count_rows_or_distinct_cases() -> None:
    """Avoid counting multiple implicates as separate survey cases."""
    df_data = pd.DataFrame(
        {
            "year": [2020, 2020, 2020, 2023, 2023],
            "case": [1, 1, 2, 1, 2],
            "value": [1.0, 2.0, 3.0, np.nan, 4.0],
        }
    )

    df_rows = plots._nobs_by_wave(
        df_data,
        "value",
        wave_column="year",
        id_column=None,
        x_column=None,
    )
    df_cases = plots._nobs_by_wave(
        df_data,
        "value",
        wave_column="year",
        id_column="case",
        x_column=None,
    )

    assert df_rows["nobs"].tolist() == [3, 1]
    assert df_cases["nobs"].tolist() == [2, 1]
    assert df_cases["x"].tolist() == [2020, 2023]


def test_wave_coordinate_can_differ_from_group_identifier() -> None:
    """Group on a wave ID while plotting its associated datetime coordinate."""
    df_data = pd.DataFrame(
        {
            "wave": [1, 1, 2, 2],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]
            ),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    df_stats, _, _ = plots._stats_by_wave(
        df_data,
        "value",
        wave_column="wave",
        x_column="date",
        weight_column=None,
        value_labels=None,
        outlier_method=plots.OutlierMethod.NONE,
        outlier_tails=plots.OutlierTail.AUTO,
        outlier_tail_fraction=0.01,
        outlier_iqr_factor=100.0,
    )

    assert df_stats["x"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]
    assert df_stats["mean"].tolist() == [1.5, 3.5]


def test_quantile_trimming_removes_only_the_applicable_tail_from_mean() -> None:
    """Trim an extreme positive value without changing full-sample quantiles."""
    values = [*np.arange(1.0, 101.0), 10_000.0]
    df_data = pd.DataFrame({"wave": 1, "value": values})

    df_stats, n_masked, tails = plots._stats_by_wave(
        df_data,
        "value",
        wave_column="wave",
        x_column=None,
        weight_column=None,
        value_labels=None,
        outlier_method=plots.OutlierMethod.QUANTILE,
        outlier_tails=plots.OutlierTail.AUTO,
        outlier_tail_fraction=0.01,
        outlier_iqr_factor=100.0,
    )

    assert tails is plots.OutlierTail.UPPER
    assert n_masked == 1
    assert df_stats.loc[0, "mean"] == pytest.approx(50.5)
    assert df_stats.loc[0, "q3"] == np.quantile(values, 0.75)


def test_tail_overrides_allow_integer_valued_magnitudes() -> None:
    """Permit explicit trimming when automatic inference exempts integers."""
    values = pd.Series([*range(1, 101), 10_000], dtype="Int64")
    df_data = pd.DataFrame({"wave": 1, "value": values})

    _, automatic_count, automatic_tails = plots._stats_by_wave(
        df_data,
        "value",
        wave_column="wave",
        x_column=None,
        weight_column=None,
        value_labels=None,
        outlier_method=plots.OutlierMethod.QUANTILE,
        outlier_tails=plots.OutlierTail.AUTO,
        outlier_tail_fraction=0.01,
        outlier_iqr_factor=100.0,
    )
    _, overridden_count, overridden_tails = plots._stats_by_wave(
        df_data,
        "value",
        wave_column="wave",
        x_column=None,
        weight_column=None,
        value_labels=None,
        outlier_method=plots.OutlierMethod.QUANTILE,
        outlier_tails={"value": plots.OutlierTail.UPPER},
        outlier_tail_fraction=0.01,
        outlier_iqr_factor=100.0,
    )

    assert automatic_tails is plots.OutlierTail.NONE
    assert automatic_count == 0
    assert overridden_tails is plots.OutlierTail.UPPER
    assert overridden_count == 1


def test_iqr_trimming_is_scale_invariant_and_ignores_zero_iqr() -> None:
    """Use relative IQR fences without introducing a unit-specific floor."""
    values = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    scaled = 1_000.0 * values

    mask = plots._mean_outlier_mask(
        values,
        None,
        method=plots.OutlierMethod.IQR,
        tails=plots.OutlierTail.UPPER,
        tail_fraction=0.01,
        iqr_factor=3.0,
    )
    scaled_mask = plots._mean_outlier_mask(
        scaled,
        None,
        method=plots.OutlierMethod.IQR,
        tails=plots.OutlierTail.UPPER,
        tail_fraction=0.01,
        iqr_factor=3.0,
    )
    zero_iqr_mask = plots._mean_outlier_mask(
        np.array([1.0, 1.0, 1.0, 100.0]),
        None,
        method=plots.OutlierMethod.IQR,
        tails=plots.OutlierTail.UPPER,
        tail_fraction=0.01,
        iqr_factor=3.0,
    )

    np.testing.assert_array_equal(mask, [False, False, False, False, True])
    np.testing.assert_array_equal(scaled_mask, mask)
    assert not zero_iqr_mask.any()


def test_grid_uses_bold_italic_suptitle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the configured title emphasis to every report."""
    figures: list[Figure] = []

    def capture_figure(fig: Figure, *_args: object, **_kwargs: object) -> None:
        figures.append(fig)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    plots._write_grid(
        ["value"],
        Path("unused.pdf"),
        lambda _ax, _variable: None,
        suptitle="Diagnostic title",
    )

    title = next(
        text for text in figures[0].texts if text.get_text() == "Diagnostic title"
    )
    assert title.get_fontweight() == plots.FIGURE_TITLE_STYLE["fontweight"]
    assert title.get_fontstyle() == plots.FIGURE_TITLE_STYLE["fontstyle"]


def test_stats_plot_draws_isolated_points_and_iqr_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw markers and vertical IQR ranges for isolated valid waves."""
    figures: list[Figure] = []

    def capture_figure(fig: Figure, *_args: object, **_kwargs: object) -> None:
        figures.append(fig)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    df_data = pd.DataFrame(
        {
            "wave": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "value": [0.0, 1.0, 8.0, np.nan, np.nan, np.nan, 10.0, 11.0, 18.0],
        }
    )

    plots.plot_stats_by_wave(
        df_data,
        ["value"],
        Path("unused.pdf"),
        wave_column="wave",
        suptitle="Statistics",
    )

    panel = figures[0].axes[1]
    isolated_lines = [
        line for line in panel.lines if line.get_marker() == plots.ISOLATED_POINT_MARKER
    ]
    assert len(isolated_lines) == 2
    np.testing.assert_allclose(
        [line.get_xdata() for line in isolated_lines],
        [[1, 3], [1, 3]],
    )
    np.testing.assert_allclose(
        [line.get_ydata() for line in isolated_lines],
        [[1, 11], [3, 13]],
    )

    iqr_lines = [
        collection
        for collection in panel.collections
        if isinstance(collection, LineCollection)
    ]
    assert len(iqr_lines) == 1
    np.testing.assert_allclose(
        iqr_lines[0].get_segments(),
        [
            [[1, 0.5], [1, 4.5]],
            [[3, 10.5], [3, 14.5]],
        ],
    )


def test_public_plot_functions_write_files_and_close_figures(tmp_path: Path) -> None:
    """Write panel and wave diagnostics without retaining Matplotlib figures."""
    df_data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "wave": [1, 2, 1, 2],
            "value": pd.Series([0, 1, 1, pd.NA], dtype="Int8"),
            "weight": [1.0, 2.0, 1.0, 2.0],
        }
    )
    paths = (
        plots.plot_nobs_by_id(
            df_data,
            ["value"],
            tmp_path / "id.pdf",
            id_column="id",
            suptitle="By ID",
        ),
        plots.plot_nobs_by_wave(
            df_data,
            ["value"],
            tmp_path / "wave.pdf",
            wave_column="wave",
            id_column="id",
            suptitle="By wave",
        ),
        plots.plot_stats_by_wave(
            df_data,
            ["value"],
            tmp_path / "stats.pdf",
            wave_column="wave",
            weight_column="weight",
            suptitle="Statistics",
        ),
    )

    assert all(path.stat().st_size > 0 for path in paths)
    assert not plt.get_fignums()
