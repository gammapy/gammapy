# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from gammapy.irf import EffectiveAreaTable2D
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.visualization.plotters.irf import EffectiveAreaPlotter


@pytest.fixture(scope="session")
def aeff():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    return EffectiveAreaTable2D.read(filename, hdu="AEFF")


@requires_data()
def test_plot(aeff):
    plotter = EffectiveAreaPlotter()

    with mpl_plot_check():
        plotter.plot(aeff)

    with mpl_plot_check():
        plotter.plot_energy_dependence(aeff)

    with mpl_plot_check():
        plotter.plot_offset_dependence(aeff)


@requires_data()
def test_peek(aeff):
    plotter = EffectiveAreaPlotter()

    with mpl_plot_check():
        plotter.peek(aeff)