# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.container import ErrorbarContainer
import astropy.units as u

from gammapy.visualization.violin import plot_flux_violin


def test_plot_flux_violin_minimal():
    """Test that plot_flux_violin runs and returns artists for a simple case."""

    energy_edges = np.array([1, 2, 4]) * u.TeV  # two bins

    samples_per_band = [
        np.array([1.0, 1.2, 0.8]) * u.Unit("cm-2 s-1 TeV-1"),
        np.array([0.4, 0.6, 0.5]) * u.Unit("cm-2 s-1 TeV-1"),
    ]

    weights_per_band = [
        np.array([1.0, 2.0, 1.0]),
        None,  # second bin: unweighted
    ]

    fig, ax = plt.subplots()

    artists = plot_flux_violin(
        ax=ax,
        energy_edges=energy_edges,
        samples_per_band=samples_per_band,
        weights_per_band=weights_per_band,
        energy_power=0.0,
        color="C0",
        alpha=0.5,
        edgecolor="black",
        lw=1.0,
        bw_method="scott",
        grid_size=50,
        violin_clip=(0.05, 0.95),
        y_label="dN/dE",
    )

    assert isinstance(artists, list)
    assert len(artists) > 0

    assert all(isinstance(a, (Artist, ErrorbarContainer)) for a in artists)

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    plt.close(fig)


def test_plot_flux_violin_empty_weights_ok():
    """Test that bins with empty or zero-sum weights are skipped safely."""

    energy_edges = np.array([1, 3]) * u.TeV
    samples_per_band = [np.array([1.0]) * u.Unit("cm-2 s-1 TeV-1")]
    weights_per_band = [np.array([0.0])]  # zero-sum → skipped internally

    fig, ax = plt.subplots()

    artists = plot_flux_violin(ax, energy_edges, samples_per_band, weights_per_band)

    assert isinstance(artists, list)
    plt.close(fig)


def test_plot_flux_violin_no_weights():
    """Test that plot_flux_violin works when no weights are provided (uniform implied)."""

    energy_edges = np.array([1.0, 2.0, 4.0]) * u.TeV

    flux_unit = u.Unit("cm-2 s-1 TeV-1")
    samples_per_band = [
        np.array([1.0, 1.2, 0.8, 1.1]) * flux_unit,  # bin 1
        np.array([0.5, 0.6, 0.55, 0.52]) * flux_unit,  # bin 2
    ]

    fig, ax = plt.subplots()

    artists = plot_flux_violin(
        ax=ax,
        energy_edges=energy_edges,
        samples_per_band=samples_per_band,
        weights_per_band=None,
        energy_power=0.0,
        color="C0",
        alpha=0.4,
        edgecolor="black",
        lw=1.0,
        bw_method="scott",
        grid_size=64,
        y_label="dN/dE",
    )

    assert isinstance(artists, list)
    assert len(artists) > 0

    assert all(isinstance(a, (Artist, ErrorbarContainer)) for a in artists)

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    assert "Energy" in ax.get_xlabel()
    assert "dN/dE" in ax.get_ylabel()

    plt.close(fig)
