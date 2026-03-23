# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from gammapy.visualization.violin import plot_samples_violin_vs_energy


def test_plot_samples_violin_vs_energy_invalid():
    with pytest.raises(ValueError, match="samples_per_band must match number of bins"):
        energy_edges = np.array([1, 2, 4]) * u.TeV

        samples_per_band = [
            np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
        ]

        _, ax = plt.subplots()
        plot_samples_violin_vs_energy(
            ax=ax,
            energy_edges=energy_edges,
            samples_per_band=samples_per_band,
        )

    with pytest.raises(ValueError, match="weights_per_band must match number of bins."):
        energy_edges = np.array([1, 2, 4]) * u.TeV

        samples_per_band = [
            np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
            np.array([0.4, 0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
        ]
        weights_per_band = [
            np.array([1.0, 2.0, 1.0]),
        ]

        _, ax = plt.subplots()
        plot_samples_violin_vs_energy(
            ax=ax,
            energy_edges=energy_edges,
            samples_per_band=samples_per_band,
            weights_per_band=weights_per_band,
        )

    with pytest.raises(
        ValueError, match="energy_edges must be 1D and contain at least two values."
    ):
        energy_edges = np.array([1]) * u.TeV

        samples_per_band = [
            np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
            np.array([0.4, 0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
        ]

        _, ax = plt.subplots()
        plot_samples_violin_vs_energy(
            ax=ax,
            energy_edges=energy_edges,
            samples_per_band=samples_per_band,
        )

    with pytest.raises(
        ValueError, match="energy_edges must be strictly increasing, finite, and > 0."
    ):
        energy_edges = np.array([-np.inf, 2, 3]) * u.TeV

        samples_per_band = [
            np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
            np.array([0.4, 0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
        ]

        _, ax = plt.subplots()
        plot_samples_violin_vs_energy(
            ax=ax,
            energy_edges=energy_edges,
            samples_per_band=samples_per_band,
        )

    with pytest.raises(
        ValueError, match="violin_clip must satisfy 0 <= low < high <= 1."
    ):
        energy_edges = np.array([1, 2, 3]) * u.TeV

        samples_per_band = [
            np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
            np.array([0.4, 0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
        ]

        _, ax = plt.subplots()
        plot_samples_violin_vs_energy(
            ax=ax,
            energy_edges=energy_edges,
            samples_per_band=samples_per_band,
            violin_clip=[10, 20],
        )


def test_plot_samples_violin_vs_energy_minimal():
    """Test that plot_samples_violin_vs_energy runs and returns artists for a simple case."""

    energy_edges = np.array([1, 2, 4]) * u.TeV

    samples_per_band = [
        np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
        np.array([-0.4, -0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
    ]

    weights_per_band = [
        np.array([1.0, 2.0, 1.0]),
        None,  # second bin: unweighted
    ]

    fig, ax = plt.subplots()

    ax = plot_samples_violin_vs_energy(
        ax=ax,
        energy_edges=energy_edges,
        samples_per_band=samples_per_band,
        weights_per_band=weights_per_band,
        bw_method="scott",
        energy_power=0.0,
        grid_size=50,
        violin_clip=(0.05, 0.95),
        y_label="dN/dE",
    )

    assert len(ax.lines) > 0

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    plt.close(fig)


def test_plot_samples_violin_vs_energy_empty_weights_ok():
    """Test that bins with empty or zero-sum weights are skipped safely."""

    energy_edges = np.array([1, 3]) * u.TeV
    samples_per_band = [np.array([1.0]) * u.Unit("cm-2 s-1 TeV-1")]
    weights_per_band = [np.array([0.0])]  # zero-sum → skipped internally

    fig, ax = plt.subplots()

    ax = plot_samples_violin_vs_energy(
        ax=ax,
        energy_edges=energy_edges,
        samples_per_band=samples_per_band,
        weights_per_band=weights_per_band,
    )

    assert len(ax.lines) == 0
    plt.close(fig)


def test_plot_samples_violin_vs_energy_no_weights():
    """Test that plot_samples_violin_vs_energy works when no weights are provided (uniform implied)."""

    energy_edges = np.array([1.0, 2.0, 4.0]) * u.TeV

    flux_unit = u.Unit("cm-2 s-1 TeV-1")
    samples_per_band = [
        np.array([1.0, 1.2, 0.8, 1.1]) * flux_unit,  # bin 1
        np.array([0.5, 0.6, 0.55, 0.52]) * flux_unit,  # bin 2
    ]

    fig, ax = plt.subplots()
    violin_kwargs = dict(
        color="C0",
        alpha=0.4,
        edgecolor="black",
        lw=1.0,
    )
    ax = plot_samples_violin_vs_energy(
        ax=ax,
        energy_edges=energy_edges,
        samples_per_band=samples_per_band,
        weights_per_band=None,
        energy_power=2,
        bw_method="scott",
        grid_size=64,
        violin_kwargs=violin_kwargs,
        y_label="dN/dE",
    )

    assert len(ax.lines) > 0

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    assert "Energy" in ax.get_xlabel()
    assert "dN/dE" in ax.get_ylabel()

    plt.close(fig)
