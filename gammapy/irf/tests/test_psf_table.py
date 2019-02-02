# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...utils.testing import assert_quantity_allclose
from ...irf import TablePSF, EnergyDependentTablePSF


class TestTablePSF:
    @staticmethod
    def test_gauss():
        # Make an example PSF for testing
        width = Angle(0.3, "deg")
        
        # containment radius for 80% containment
        radius = width * np.sqrt(2 * np.log(5))
        
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="gauss", width=width, rad=rad)
        
        assert_allclose(psf.containment(radius), 0.8, rtol=1e-2)

        desired = radius.to_value("deg")
        actual = psf.containment_radius(0.8).to_value("deg")
        assert_allclose(actual, desired, rtol=1e-2)

    @staticmethod
    def test_disk():
        width = Angle(2, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="disk", width=width, rad=rad)

        psf_value = psf.evaluate(rad)
        psf_value = (2 * np.pi * rad * psf_value).to("radian^-1")
        integral = np.sum(np.diff(rad.radian) * psf_value[:-1])

        assert_allclose(integral.value, 1, rtol=1e-3)
        assert_allclose(psf.containment(Angle(2, "deg")), 1, rtol=1e-3)


    # TODO: is this useful in addition to the previous tests?
    @staticmethod
    def test_more():
        # Make an example PSF for testing
        width = Angle(0.3, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="gauss", width=width, rad=rad)

        # Test inputs
        rad = Angle([0.1, 0.3], "deg")

        actual = psf.evaluate(rad=rad)
        desired = u.Quantity([5491.52067694, 3521.07804604], "sr^-1")
        assert_quantity_allclose(actual, desired)

        rad_min = Angle([0.0, 0.1, 0.3], "deg")
        rad_max = Angle([0.1, 0.3, 2.0], "deg")
        actual = psf.containment(rad_max) - psf.containment(rad_min)
        desired = [0.055256, 0.340536, 0.604203]
        assert_allclose(actual, desired, rtol=1e-5)


@requires_data("gammapy-data")
class TestEnergyDependentTablePSF:
    def setup(self):
        filename = "$GAMMAPY_DATA/tests/unbundled/fermi/psf.fits"
        self.psf = EnergyDependentTablePSF.read(filename)

    def test(self):
        # TODO: test __init__

        # Test cases
        energy = u.Quantity(1, "GeV")
        rad = Angle(0.1, "deg")
        energies = u.Quantity([1, 2], "GeV").to("TeV")
        rads = Angle([0.1, 0.2], "deg")

        # actual = psf.evaluate(energy=energy, rad=rad)
        # desired = u.Quantity(17760.814249206363, 'sr^-1')
        # assert_quantity_allclose(actual, desired)

        # actual = psf.evaluate(energy=energies, rad=rads)
        # desired = u.Quantity([17760.81424921, 5134.17706619], 'sr^-1')
        # assert_quantity_allclose(actual, desired)

        psf1 = self.psf.table_psf_at_energy(energy)
        containment = np.linspace(0, 0.95, 3)
        actual = psf1.containment_radius(containment).to_value("deg")
        desired =  [0., 0.188798, 1.026798]
        assert_allclose(actual, desired, rtol=1e-5)
        # TODO: test average_psf
        # psf2 = psf.psf_in_energy_band(energy_band, spectrum)

        # TODO: test containment_radius
        # TODO: test containment_fraction
        # TODO: test info
        # TODO: test plotting methods

        energy_band = u.Quantity([10, 500], "GeV")
        psf_band = self.psf.table_psf_in_energy_band(energy_band)

    @requires_dependency("matplotlib")
    def test_plot(self):
        with mpl_plot_check():
            self.psf.plot_containment_vs_energy()

        energy = u.Quantity(1, "GeV")
        psf_1GeV = self.psf.table_psf_at_energy(energy)
        with mpl_plot_check():
            psf_1GeV.plot_psf_vs_rad()

    @requires_dependency("matplotlib")
    def test_plot2(self):
        # psf.plot_containment('fermi_psf_containment.pdf')
        # psf.plot_exposure('fermi_psf_exposure.pdf')
        with mpl_plot_check():
            self.psf.plot_psf_vs_rad()

    def test_repr(self):
        info = str(self.psf)
        assert "Containment" in info