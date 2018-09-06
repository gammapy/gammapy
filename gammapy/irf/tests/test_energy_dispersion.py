# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...utils.energy import EnergyBounds
from ...irf import EnergyDispersion, EnergyDispersion2D


@requires_dependency("scipy")
class TestEnergyDispersion:
    def setup(self):
        self.e_true = np.logspace(0, 1, 101) * u.TeV
        self.e_reco = self.e_true
        self.resolution = 0.1
        self.bias = 0
        self.edisp = EnergyDispersion.from_gauss(
            e_true=self.e_true,
            e_reco=self.e_reco,
            pdf_threshold=1e-7,
            sigma=self.resolution,
            bias=self.bias,
        )

    def test_from_diagonal_response(self):
        e_true = [0.5, 1, 2, 4, 6] * u.TeV
        e_reco = [2, 4, 6] * u.TeV

        edisp = EnergyDispersion.from_diagonal_response(e_true, e_reco)

        assert edisp.pdf_matrix.shape == (4, 2)
        expected = [[0, 0], [0, 0], [1, 0], [0, 1]]

        assert_equal(edisp.pdf_matrix, expected)

        # Test square matrix
        edisp = EnergyDispersion.from_diagonal_response(e_true)
        assert_allclose(edisp.e_reco.bins.value, e_true.value)
        assert edisp.e_reco.bins.unit == "TeV"
        assert_equal(edisp.pdf_matrix[0][0], 1)
        assert_equal(edisp.pdf_matrix[2][0], 0)
        assert edisp.pdf_matrix.sum() == 4

    def test_str(self):
        assert "EnergyDispersion" in str(self.edisp)

    def test_evaluate(self):
        # Check for correct normalization
        pdf = self.edisp.data.evaluate(e_true=3.34 * u.TeV)
        assert_allclose(np.sum(pdf), 1, atol=1e-2)

    def test_apply(self):
        counts = np.arange(len(self.e_true) - 1)
        actual = self.edisp.apply(counts)
        assert_allclose(actual[0], 1.8612999017723058, atol=1e-3)

        counts = np.arange(len(self.e_true) - 4)
        with pytest.raises(ValueError) as exc:
            self.edisp.apply(counts)
        assert str(len(counts)) in str(exc.value)
        assert_allclose(actual[0], 1.8612999017723058, atol=1e-3)

    def test_get_bias(self):
        bias = self.edisp.get_bias(3.34 * u.TeV)
        assert_allclose(bias, self.bias, atol=1e-2)

    def test_get_resolution(self):
        resolution = self.edisp.get_resolution(3.34 * u.TeV)
        assert_allclose(resolution, self.resolution, atol=1e-2)

    def test_io(self, tmpdir):
        indices = np.array([[1, 3, 6], [3, 3, 2]])
        desired = self.edisp.pdf_matrix[indices]
        writename = str(tmpdir / "rmf_test.fits")
        self.edisp.write(writename)
        edisp2 = EnergyDispersion.read(writename)
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


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestEnergyDispersion2D:
    def setup(self):
        # TODO: use from_gauss method to create know edisp (see below)
        # At the moment only 1 test uses it (test_get_response)
        filename = (
            "$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz"
        )
        self.edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")

        # Make a test case
        e_true = np.logspace(-1., 2., 51) * u.TeV
        migra = np.linspace(0., 4., 1001)
        offset = np.linspace(0., 2.5, 5) * u.deg
        sigma = 0.15 / (e_true[:-1] / (1 * u.TeV)).value ** 0.3
        bias = 1e-3 * (e_true[:-1] - 1 * u.TeV).value
        self.edisp2 = EnergyDispersion2D.from_gauss(e_true, migra, bias, sigma, offset)

    def test_str(self):
        assert "EnergyDispersion2D" in str(self.edisp)

    def test_evaluation(self):
        # TODO: Move to tests for NDDataArray
        # Check that nodes are evaluated correctly
        e_node = 12
        off_node = 3
        m_node = 5
        offset = self.edisp.data.axis("offset").nodes[off_node]
        energy = self.edisp.data.axis("e_true").nodes[e_node]
        migra = self.edisp.data.axis("migra").nodes[m_node]
        actual = self.edisp.data.evaluate(offset=offset, e_true=energy, migra=migra)
        desired = self.edisp.data.data[e_node, m_node, off_node]
        assert_allclose(actual, desired, rtol=1e-06)
        assert_allclose(actual, 0.09388659149, rtol=1e-06)

        # Check output shape
        energy = [1, 2] * u.TeV
        migra = [0.98, 0.97, 0.7]
        offset = [0.1, 0.2] * u.deg
        actual = self.edisp.data.evaluate(e_true=energy, migra=migra, offset=offset)
        assert_allclose(actual.shape, (2, 3, 2))

        # Check evaluation at all nodes
        actual = self.edisp.data.evaluate().shape
        desired = (
            self.edisp.data.axis("e_true").nbins,
            self.edisp.data.axis("migra").nbins,
            self.edisp.data.axis("offset").nbins,
        )
        assert_equal(actual, desired)

    def test_get_response(self):
        pdf = self.edisp2.get_response(offset=0.7 * u.deg, e_true=1 * u.TeV)
        assert_allclose(pdf.sum(), 1)
        assert_allclose(pdf.max(), 0.013025634736094305)

    def test_exporter(self):
        # Check RMF exporter
        offset = Angle(0.612, "deg")
        e_reco = EnergyBounds.equal_log_spacing(1, 10, 6, "TeV")
        e_true = EnergyBounds.equal_log_spacing(0.8, 5, 4, "TeV")
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
        assert_equal(hdu.data["ENERG_LO"][0], edisp.data.axis("e_true").lo.value)
        assert hdu.header["TUNIT1"] == edisp.data.axis("e_true").lo.unit

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
