# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle
from gammapy.irf import EnergyDependentTablePSF, TablePSF, PSF3D
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture(scope="session")
def psf_3d():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    return PSF3D.read(filename, hdu="PSF")


@requires_data()
def test_psf_3d_basics(psf_3d):
    rad_axis = psf_3d.axes["rad"]
    assert_allclose(rad_axis.edges[-2].value, 0.659048, rtol=1e-5)
    assert rad_axis.nbin == 144
    assert rad_axis.unit == "deg"

    energy_axis_true = psf_3d.axes["energy_true"]
    assert_allclose(energy_axis_true.edges[0].value, 0.01)
    assert energy_axis_true.nbin == 32
    assert energy_axis_true.unit == "TeV"

    assert psf_3d.data.shape == (32, 6, 144)
    assert psf_3d.unit == "sr-1"

    assert_allclose(psf_3d.meta["LO_THRES"], 0.01)

    assert "PSF3D" in str(psf_3d)

    with pytest.raises(ValueError):
        PSF3D(axes=psf_3d.axes, data=psf_3d.data.T)


@requires_data()
def test_psf_3d_evaluate(psf_3d):
    q = psf_3d.evaluate(energy_true="1 TeV", offset="0.3 deg", rad="0.1 deg")
    assert_allclose(q.value, 25847.249548)
    # TODO: is this the shape we want here?
    assert q.shape == ()
    assert q.unit == "sr-1"


@requires_data()
def test_to_energy_dependent_table_psf(psf_3d):
    psf = psf_3d.to_energy_dependent_table_psf(offset=0 * u.deg)
    assert psf.data.data.shape == (32, 66)
    radius = psf.table_psf_at_energy("1 TeV").containment_radius(0.68)
    assert_allclose(radius, 0.123352 * u.deg, atol=1e-2)


@requires_data()
def test_psf_3d_containment_radius(psf_3d):
    q = psf_3d.containment_radius(
        energy_true=1 * u.TeV, fraction=0.68, offset=0 * u.deg
    )
    assert_allclose(q.value, 0.123352, rtol=1e-2)
    assert q.isscalar
    assert q.unit == "deg"

    q = psf_3d.containment_radius(
        energy_true=[1, 3] * u.TeV, fraction=0.68, offset=0 * u.deg
    )
    assert_allclose(q.value, [0.123261, 0.13131], rtol=1e-2)
    assert q.shape == (2,)


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_plot_vs_rad(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_psf_vs_rad()


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_plot_containment(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_containment_radius()


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_peek(psf_3d):
    with mpl_plot_check():
        psf_3d.peek()


class TestTablePSF:
    @staticmethod
    def test_gauss():
        # Make an example PSF for testing
        width = Angle(0.3, "deg")

        # containment radius for 80% containment
        radius = width * np.sqrt(2 * np.log(5))

        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="gauss", width=width, rad=rad)

        assert_allclose(psf.containment(radius), 0.8, rtol=1e-4)

        desired = radius.to_value("deg")
        actual = psf.containment_radius(0.8).to_value("deg")
        assert_allclose(actual, desired, rtol=1e-4)

    @staticmethod
    def test_disk():
        width = Angle(2, "deg")
        rad = Angle(np.linspace(0, 2.3, 1000), "deg")
        psf = TablePSF.from_shape(shape="disk", width=width, rad=rad)

        # test containment
        radius = Angle(1, "deg")
        actual = psf.containment(radius)
        desired = (radius / width).to_value("") ** 2
        assert_allclose(actual, desired, rtol=1e-4)

        # test containment radius
        actual = psf.containment_radius(0.25)
        assert_allclose(actual, radius, rtol=1e-4)

        # test info
        info = psf.info()
        assert info.find("integral") == 66


@requires_data()
class TestEnergyDependentTablePSF:
    def setup(self):
        filename = "$GAMMAPY_DATA/tests/unbundled/fermi/psf.fits"
        self.psf = EnergyDependentTablePSF.read(filename)

    def test(self):
        # TODO: test __init__

        # Test cases
        energy = u.Quantity(1, "GeV")

        psf1 = self.psf.table_psf_at_energy(energy)
        containment = np.linspace(0, 0.95, 3)
        actual = psf1.containment_radius(containment).to_value("deg")
        desired = [0., 0.251731, 0.967178]
        assert_allclose(actual, desired, rtol=1e-2)

        # TODO: test average_psf
        # TODO: test containment_radius
        # TODO: test containment_fraction
        # TODO: test info
        # TODO: test plotting methods

        energy_range = u.Quantity([10, 500], "GeV")
        psf_band = self.psf.table_psf_in_energy_range(energy_range)
        # TODO: add assert

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
        with mpl_plot_check():
            self.psf.plot_psf_vs_rad()

    @requires_dependency("matplotlib")
    def test_plot_exposure_vs_energy(self):
        with mpl_plot_check():
            self.psf.plot_exposure_vs_energy()

    def test_write(self, tmp_path):
        self.psf.write(tmp_path / "test.fits")
        new = EnergyDependentTablePSF.read(tmp_path / "test.fits")
        assert_allclose(new.axes["rad"].center, self.psf.axes["rad"].center)
        assert_allclose(new.axes["energy_true"].center, self.psf.axes["energy_true"].center)
        assert_allclose(new.quantity, self.psf.quantity)

    def test_repr(self):
        info = str(self.psf)
        assert "EnergyDependentTablePSF" in info
