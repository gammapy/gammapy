# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import EDispKernel, EnergyDispersion2D
from gammapy.maps import MapAxis
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


class TestEDispKernel:
    def setup(self):
        self.e_true = np.logspace(0, 1, 101) * u.TeV
        self.e_reco = self.e_true
        self.resolution = 0.1
        self.bias = 0
        self.edisp = EDispKernel.from_gauss(
            e_true=self.e_true,
            e_reco=self.e_reco,
            pdf_threshold=1e-7,
            sigma=self.resolution,
            bias=self.bias,
        )

    def test_from_diagonal_response(self):
        e_true = [0.5, 1, 2, 4, 6] * u.TeV
        e_reco = [2, 4, 6] * u.TeV

        edisp = EDispKernel.from_diagonal_response(e_true, e_reco)

        assert edisp.pdf_matrix.shape == (4, 2)
        expected = [[0, 0], [0, 0], [1, 0], [0, 1]]

        assert_equal(edisp.pdf_matrix, expected)

        # Test square matrix
        edisp = EDispKernel.from_diagonal_response(e_true)
        assert_allclose(edisp.e_reco.edges.value, e_true.value)
        assert edisp.e_reco.unit == "TeV"
        assert_equal(edisp.pdf_matrix[0][0], 1)
        assert_equal(edisp.pdf_matrix[2][0], 0)
        assert edisp.pdf_matrix.sum() == 4

    def test_str(self):
        assert "EDispKernel" in str(self.edisp)

    def test_evaluate(self):
        # Check for correct normalization
        pdf = self.edisp.data.evaluate(energy_true=3.34 * u.TeV)
        assert_allclose(np.sum(pdf), 1, atol=1e-2)

    def test_get_bias(self):
        bias = self.edisp.get_bias(3.34 * u.TeV)
        assert_allclose(bias, self.bias, atol=1e-2)

    def test_get_resolution(self):
        resolution = self.edisp.get_resolution(3.34 * u.TeV)
        assert_allclose(resolution, self.resolution, atol=1e-2)

    def test_io(self, tmp_path):
        indices = np.array([[1, 3, 6], [3, 3, 2]])
        desired = self.edisp.pdf_matrix[indices]
        self.edisp.write(tmp_path / "tmp.fits")
        edisp2 = EDispKernel.read(tmp_path / "tmp.fits")
        actual = edisp2.pdf_matrix[indices]
        assert_allclose(actual, desired)

    @requires_dependency("matplotlib")
    def test_plot_matrix(self):
        with mpl_plot_check():
            self.edisp.plot_matrix()

    @requires_dependency("matplotlib")
    def test_plot_bias(self):
        with mpl_plot_check():
            self.edisp.plot_bias()

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.edisp.peek()


@requires_data()
class TestEnergyDispersion2D:
    @classmethod
    def setup_class(cls):
        filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
        cls.edisp = EnergyDispersion2D.read(filename, hdu="EDISP")

        # Make a test case
        e_true = np.logspace(-1.0, 2.0, 51) * u.TeV
        migra = np.linspace(0.0, 4.0, 1001)
        offset = np.linspace(0.0, 2.5, 5) * u.deg
        sigma = 0.15 / (e_true[:-1] / (1 * u.TeV)).value ** 0.3
        bias = 1e-3 * (e_true[:-1] - 1 * u.TeV).value
        cls.edisp2 = EnergyDispersion2D.from_gauss(e_true, migra, bias, sigma, offset)

    def test_str(self):
        assert "EnergyDispersion2D" in str(self.edisp)

    def test_evaluation(self):
        # Check output shape
        energy = [1, 2] * u.TeV
        migra = np.array([0.98, 0.97, 0.7])
        offset = [0.1, 0.2, 0.3, 0.4] * u.deg
        actual = self.edisp.data.evaluate(
            energy_true=energy.reshape(-1, 1, 1),
            migra=migra.reshape(1, -1, 1),
            offset=offset.reshape(1, 1, -1),
        )
        assert_allclose(actual.shape, (2, 3, 4))

        # Check evaluation at all nodes
        actual = self.edisp.data.evaluate().shape
        desired = (
            self.edisp.data.axis("energy_true").nbin,
            self.edisp.data.axis("migra").nbin,
            self.edisp.data.axis("offset").nbin,
        )
        assert_equal(actual, desired)

    def test_get_response(self):
        pdf = self.edisp2.get_response(offset=0.7 * u.deg, e_true=1 * u.TeV)
        assert_allclose(pdf.sum(), 1)
        assert_allclose(pdf.max(), 0.010421, rtol=1e-4)

    def test_exporter(self):
        # Check RMF exporter
        offset = Angle(0.612, "deg")
        e_reco = MapAxis.from_energy_bounds(1, 10, 7, "TeV").edges
        e_true = MapAxis.from_energy_bounds(0.8, 5, 5, "TeV").edges
        rmf = self.edisp.to_energy_dispersion(offset, e_true=e_true, e_reco=e_reco)
        assert_allclose(rmf.data.data[2, 3], 0.08, atol=5e-2)  # same tolerance as above
        actual = rmf.pdf_matrix[2]
        e_val = np.sqrt(e_true[2] * e_true[3])
        desired = self.edisp.get_response(offset, e_val, e_reco)
        assert_equal(actual, desired)

    def test_write(self):
        energy_lo = np.logspace(0, 1, 11)[:-1] * u.TeV
        energy_hi = np.logspace(0, 1, 11)[1:] * u.TeV
        offset_lo = np.linspace(0, 1, 4)[:-1] * u.deg
        offset_hi = np.linspace(0, 1, 4)[1:] * u.deg
        migra_lo = np.linspace(0, 3, 4)[:-1]
        migra_hi = np.linspace(0, 3, 4)[1:]

        data = (
            np.ones(shape=(len(energy_lo), len(migra_lo), len(offset_lo))) * u.cm * u.cm
        )

        edisp = EnergyDispersion2D(
            e_true_lo=energy_lo,
            e_true_hi=energy_hi,
            migra_lo=migra_lo,
            migra_hi=migra_hi,
            offset_lo=offset_lo,
            offset_hi=offset_hi,
            data=data,
        )

        hdu = edisp.to_fits()
        energy = edisp.data.axis("energy_true").edges
        assert_equal(hdu.data["ENERG_LO"][0], energy[:-1].value)
        assert hdu.header["TUNIT1"] == edisp.data.axis("energy_true").unit

    @requires_dependency("matplotlib")
    def test_plot_migration(self):
        with mpl_plot_check():
            self.edisp.plot_migration()

    @requires_dependency("matplotlib")
    def test_plot_bias(self):
        with mpl_plot_check():
            self.edisp.plot_bias()

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.edisp.peek()


@requires_data("gammapy-data")
def test_get_bias_energy():
    """Obs read from file"""
    rmffile = "$GAMMAPY_DATA/joint-crab/spectra/hess/rmf_obs23523.fits"
    edisp = EDispKernel.read(rmffile)
    thresh_lo = edisp.get_bias_energy(0.1)
    assert_allclose(thresh_lo.to("TeV").value, 0.9174, rtol=1e-4)
