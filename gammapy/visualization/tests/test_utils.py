# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from scipy.stats import norm
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from gammapy.maps import Map, MapAxis, WcsNDMap
from gammapy.utils.random import get_random_state
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.visualization import (
    add_colorbar,
    plot_contour_line,
    plot_distribution,
    plot_map_rgb,
    plot_theta_squared_table,
)


@requires_data()
def test_add_colorbar():
    map_ = Map.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")

    fig, ax = plt.subplots()
    with mpl_plot_check():
        img = ax.imshow(map_.sum_over_axes().data[0, :, :])
        add_colorbar(img, ax=ax, label="Colorbar label")

    axes_loc = {"position": "left", "size": "2%", "pad": "15%"}
    fig, ax = plt.subplots()
    with mpl_plot_check():
        img = ax.imshow(map_.sum_over_axes().data[0, :, :])
        add_colorbar(img, ax=ax, axes_loc=axes_loc)

    kwargs = {"use_gridspec": False, "orientation": "horizontal"}
    fig, ax = plt.subplots()
    with mpl_plot_check():
        img = ax.imshow(map_.sum_over_axes().data[0, :, :])
        cbar = add_colorbar(img, ax=ax, **kwargs)
        assert cbar.orientation == "horizontal"


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
def test_plot_map_rgb():
    map_ = Map.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")

    with pytest.raises(ValueError):
        plot_map_rgb(map_)

    with pytest.raises(ValueError):
        plot_map_rgb(map_.sum_over_axes(keepdims=False))

    axis = MapAxis([0, 1, 2, 3], node_type="edges")
    map_allsky = WcsNDMap.create(binsz=10 * u.deg, axes=[axis])
    with mpl_plot_check():
        plot_map_rgb(map_allsky)

    axis_rgb = MapAxis.from_energy_edges(
        [0.1, 0.2, 0.5, 10], unit=u.TeV, name="energy", interp="log"
    )
    map_ = map_.resample_axis(axis_rgb)
    kwargs = {"stretch": 0.5, "Q": 1, "minimum": 0.15}
    with mpl_plot_check():
        plot_map_rgb(map_, **kwargs)


def test_plot_distribution():
    random_state = get_random_state(0)
    array = random_state.normal(0, 1, 10000)

    array_2d = array.reshape(1, 100, 100)

    energy_axis = MapAxis.from_energy_edges([1, 10] * u.TeV)

    map_ = WcsNDMap.create(npix=(100, 100), axes=[energy_axis])
    map_.data = array_2d

    energy_axis_10 = MapAxis.from_energy_bounds(1 * u.TeV, 10 * u.TeV, 10)
    map_empty = WcsNDMap.create(npix=(100, 100), axes=[energy_axis_10])

    def fit_func(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    with mpl_plot_check():
        axes, res = plot_distribution(
            wcs_map=map_, func=fit_func, kwargs_hist={"bins": 40}
        )

        assert axes.shape == (1,)
        assert "info_dict" in res[0]
        assert ["fvec", "nfev", "fjac", "ipvt", "qtf"] == list(
            (res[0].get("info_dict").keys())
        )
        assert res[0].get("param") is not None
        assert res[0].get("covar") is not None

        axes, res = plot_distribution(map_empty)

        assert res == []
        assert axes.shape == (4, 3)

        axes, res = plot_distribution(
            wcs_map=map_, func="norm", kwargs_hist={"bins": 40}
        )

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_distribution(map_empty, ax=ax)
