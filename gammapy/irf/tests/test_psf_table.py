# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.units import Quantity
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...utils.testing import assert_quantity_allclose
from ...irf import TablePSF, EnergyDependentTablePSF

pytest.importorskip("scipy")


class TestTablePSF:
    @staticmethod
    def test_gauss():
        # Make an example PSF for testing
        width = Angle(0.3, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="gauss", width=width, rad=rad)
        assert_allclose(psf.integral(), 1, rtol=1e-3)

    @staticmethod
    def test_disk():
        width = Angle(2, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="disk", width=width, rad=rad)

        # Check psf.evaluate by checking if probabilities sum to 1
        psf_value = psf.evaluate(rad, quantity="dp_dr")
        integral = np.sum(np.diff(rad.radian) * psf_value[:-1])
        assert_allclose(integral.value, 1, rtol=1e-3)

        psf_value = psf.evaluate(rad, quantity="dp_domega")
        psf_value = (2 * np.pi * rad * psf_value).to("radian^-1")
        integral = np.sum(np.diff(rad.radian) * psf_value[:-1])
        assert_allclose(integral.value, 1, rtol=1e-3)

        assert_allclose(psf.integral(), 1, rtol=1e-3)
        assert_allclose(psf.integral(*Angle([0, 10], "deg")), 1, rtol=1e-3)
        assert_allclose(psf.integral(*Angle([0, 1], "deg")), 0.25, rtol=1e-4)
        assert_allclose(psf.integral(*Angle([1, 2], "deg")), 0.75, rtol=1e-2)

        # TODO
        # actual = psf.containment_radius([0.01, 0.25, 0.99])
        # desired = Angle([0, 1, 2], 'deg')
        # assert_quantity_allclose(actual, desired, rtol=1e-3)

    # TODO: is this useful in addition to the previous tests?
    @staticmethod
    def test_more():
        # Make an example PSF for testing
        width = Angle(0.3, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="gauss", width=width, rad=rad)

        # Test inputs
        rad = Angle([0.1, 0.3], "deg")

        actual = psf.evaluate(rad=rad, quantity="dp_domega")
        desired = Quantity([5491.52067694, 3521.07804604], "sr^-1")
        assert_quantity_allclose(actual, desired)

        actual = psf.evaluate(rad=rad, quantity="dp_dr")
        desired = Quantity([60.22039017, 115.83738017], "rad^-1")
        assert_quantity_allclose(actual, desired, rtol=1e-6)

        rad_min = Angle([0.0, 0.1, 0.3], "deg")
        rad_max = Angle([0.1, 0.3, 2.0], "deg")
        actual = psf.integral(rad_min, rad_max)
        desired = [0.05403975, 0.33942469, 0.60653066]
        assert_allclose(actual, desired)


@requires_data("gammapy-extra")
class TestEnergyDependentTablePSF:
    def setup(self):
        filename = "$GAMMAPY_EXTRA/test_datasets/unbundled/fermi/psf.fits"
        self.psf = EnergyDependentTablePSF.read(filename)

    def test(self):
        # TODO: test __init__

        # Test cases
        energy = Quantity(1, "GeV")
        rad = Angle(0.1, "deg")
        energies = Quantity([1, 2], "GeV").to("TeV")
        rads = Angle([0.1, 0.2], "deg")

        # actual = psf.evaluate(energy=energy, rad=rad)
        # desired = Quantity(17760.814249206363, 'sr^-1')
        # assert_quantity_allclose(actual, desired)

        # actual = psf.evaluate(energy=energies, rad=rads)
        # desired = Quantity([17760.81424921, 5134.17706619], 'sr^-1')
        # assert_quantity_allclose(actual, desired)

        psf1 = self.psf.table_psf_at_energy(energy)
        containment = np.linspace(0, 0.95, 3)
        containment_radius = psf1.containment_radius(containment)
        assert_allclose(containment_radius, Angle([0, 0.19426847, 1.03806372], "deg"))
        # TODO: test average_psf
        # psf2 = psf.psf_in_energy_band(energy_band, spectrum)

        # TODO: test containment_radius
        # TODO: test containment_fraction
        # TODO: test info
        # TODO: test plotting methods

        desired = 1.0
        energy_band = Quantity([10, 500], "GeV")
        psf_band = self.psf.table_psf_in_energy_band(energy_band)

    @requires_dependency("matplotlib")
    def test_plot(self):
        with mpl_plot_check():
            self.psf.plot_containment_vs_energy()

        energy = Quantity(1, "GeV")
        psf_1GeV = self.psf.table_psf_at_energy(energy)
        with mpl_plot_check():
            psf_1GeV.plot_psf_vs_rad()

    # TODO: fix this test (move the code from examples/plot_irfs.py here)
    @pytest.mark.xfail
    @requires_dependency("matplotlib")
    def test_plot2(self):
        # psf.plot_containment('fermi_psf_containment.pdf')
        # psf.plot_exposure('fermi_psf_exposure.pdf')
        with mpl_plot_check():
            self.psf.plot_psf_vs_rad()
