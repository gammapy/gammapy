# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt
from packaging import version
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.maps import Map
from gammapy.visualization import (
    plot_contour_line,
    plot_rgb,
    plot_spectrum_datasets_off_regions,
    plot_theta_squared_table,
)


@pytest.mark.skipif(
    version.parse(matplotlib.__version__) < version.parse("3.5"),
    reason="Requires matplotlib 3.5 or higher",
)
def test_plot_spectrum_datasets_off_regions():
    from gammapy.datasets import SpectrumDatasetOnOff
    from gammapy.maps import Map, RegionNDMap

    counts_off_1 = RegionNDMap.create("icrs;circle(0, 0.5, 0.2);circle(0.5, 0, 0.2)")

    counts_off_2 = RegionNDMap.create("icrs;circle(0.5, 0.5, 0.2);circle(0, 0, 0.2)")

    counts_off_3 = RegionNDMap.create("icrs;point(0.5, 0.5);point(0, 0)")

    m = Map.from_geom(geom=counts_off_1.geom.to_wcs_geom())
    ax = m.plot()

    dataset_1 = SpectrumDatasetOnOff(counts_off=counts_off_1)

    dataset_2 = SpectrumDatasetOnOff(counts_off=counts_off_2)

    dataset_3 = SpectrumDatasetOnOff(counts_off=counts_off_3)

    plot_spectrum_datasets_off_regions(
        ax=ax, datasets=[dataset_1, dataset_2, dataset_3]
    )

    actual = ax.patches[0].get_edgecolor()
    assert_allclose(actual, (0.121569, 0.466667, 0.705882, 1.0), rtol=1e-2)

    actual = ax.patches[2].get_edgecolor()
    assert_allclose(actual, (1.0, 0.498039, 0.054902, 1.0), rtol=1e-2)
    assert ax.lines[0].get_color() in ["green", "C0"]


def test_map_panel_plotter():
    t = np.linspace(0.0, 6.1, 10)
    x = np.cos(t)
    y = np.sin(t)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    with mpl_plot_check():
        plot_contour_line(ax, x, y)

    x = np.append(x, x[0])
    y = np.append(y, y[0])
    with mpl_plot_check():
        plot_contour_line(ax, x, y)


def test_plot_theta2_distribution():
    table = Table()
    table["theta2_min"] = [0, 0.1]
    table["theta2_max"] = [0.1, 0.2]

    for column in [
        "counts",
        "counts_off",
        "excess",
        "excess_errp",
        "excess_errn",
        "sqrt_ts",
    ]:
        table[column] = [1, 1]

    # open a new figure to avoid
    plt.figure()
    plot_theta_squared_table(table=table)


@requires_data()
def test_plot_rgb():
    map_ = Map.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    kwargs = {"stretch": 0.5, "Q": 1, "minimum": 0.15}
    with mpl_plot_check():
        plot_rgb(map_, [0.1, 0.2, 0.5, 10] * u.TeV, **kwargs)

    with pytest.raises(ValueError):
        plot_rgb(map_, [0.1, 0.2, 10] * u.TeV, **kwargs)
