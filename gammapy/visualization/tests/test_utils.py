# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from gammapy.datasets import MapDataset
from gammapy.estimators import TSMapEstimator
from gammapy.maps import Map, MapAxis, WcsNDMap
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.visualization import (
    plot_contour_line,
    plot_distribution,
    plot_map_rgb,
    plot_theta_squared_table,
)


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


@requires_data()
def test_plot_distribution():
    dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")

    tsme = TSMapEstimator().run(dataset)

    with pytest.raises(ValueError):
        plot_distribution(tsme.sqrt_ts, fit=True)

    with mpl_plot_check():
        res, ax = plot_distribution(dataset.counts, fit=False)

        assert ax.shape == (4, 3)

        def func(x, mu, sig):
            return norm.pdf(x, mu, sig)

        res, ax = plot_distribution(
            tsme.sqrt_ts,
            fit=True,
            func=func,
            kwargs_hist={"bins": 40, "range": (-5, 10)},
        )

        assert len(ax) == 1
        assert len(res[0]) == 3
        assert len(res[0].get("info_dict")) == 5

        assert_allclose(res[0].get("param"), np.array([0.38355733, 1.21281365]))
        assert_allclose(
            res[0].get("covar"),
            np.array(
                [[6.04570694e-04, -1.68336683e-11], [-1.68336683e-11, 4.03047159e-04]]
            ),
        )
        assert_allclose(res[0].get("info_dict").get("nfev"), 22)
