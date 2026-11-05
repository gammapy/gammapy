# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib
import matplotlib.pyplot as plt
import pytest

from gammapy.irf import EDispKernel
from gammapy.maps import MapAxis
from gammapy.utils.testing import mpl_plot_check
from gammapy.visualization.plotters.irf import EDispKernelPlotter


@pytest.fixture(scope="module")
def kernel():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    energy_axis_true = energy_axis.copy(name="energy_true")
    return EDispKernel.from_gauss(
        energy_axis_true=energy_axis_true,
        energy_axis=energy_axis,
        sigma=0.1,
        bias=0,
    )


def test_plot_matrix(kernel):
    plotter = EDispKernelPlotter()
    with mpl_plot_check():
        ax = plotter.plot_matrix(kernel)
    assert ax is not None


def test_plot_matrix_colorbar(kernel):
    plotter = EDispKernelPlotter()
    with mpl_plot_check():
        plotter.plot_matrix(kernel, add_cbar=True)


def test_plot_bias(kernel):
    plotter = EDispKernelPlotter()
    with mpl_plot_check():
        ax = plotter.plot_bias(kernel)
    assert ax is not None
    assert ax.get_xlabel() == "$E_\\mathrm{True}$ [$\\mathrm{TeV}$]"


def test_peek(kernel):
    plotter = EDispKernelPlotter()
    with mpl_plot_check():
        plotter.peek(kernel)
