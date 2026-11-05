# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib
import matplotlib.pyplot as plt
import pytest
import astropy.units as u

from gammapy.irf import EnergyDispersion2D
from gammapy.maps import MapAxis
from gammapy.utils.testing import mpl_plot_check
from gammapy.visualization.plotters.irf import EDispPlotter


@pytest.fixture(scope="module")
def edisp():
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=10, name="energy_true"
    )
    migra_axis = MapAxis.from_bounds(0, 3, nbin=10, name="migra", node_type="edges")
    offset_axis = MapAxis.from_bounds(0, 1, nbin=2, unit="deg", name="offset")
    return EnergyDispersion2D.from_gauss(
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        offset_axis=offset_axis,
        bias=0,
        sigma=0.1,
    )


def test_plot_migration(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        ax = plotter.plot_migration(edisp)
    assert ax is not None
    assert ax.get_ylabel() == "Probability density"


def test_plot_migration_custom(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        plotter.plot_migration(edisp, offset=0 * u.deg, energy_true=1 * u.TeV)


def test_plot_bias(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        ax = plotter.plot_bias(edisp)
    assert ax is not None


def test_plot_bias_colorbar(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        plotter.plot_bias(edisp, add_cbar=True)


def test_plot_bias_custom(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        ax = plotter.plot_bias(edisp, offset=0 * u.deg)
    assert ax is not None


def test_peek(edisp):
    plotter = EDispPlotter()
    with mpl_plot_check():
        plotter.peek(edisp)
