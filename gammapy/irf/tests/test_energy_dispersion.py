# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds
from ...irf import EnergyDispersion, EnergyDispersion2D


@requires_dependency('scipy')
class TestEnergyDispersion:
    def setup(self):
        self.e_true = np.logspace(0, 1, 101) * u.TeV
        self.e_reco = self.e_true
        self.resolution = 0.1
        self.edisp = EnergyDispersion.from_gauss(e_true=self.e_true,
                                                 e_reco=self.e_reco,
                                                 pdf_threshold=1e-7,
                                                 sigma=self.resolution)

    def test_basic(self):
        assert 'EnergyDispersion' in str(self.edisp)
        test_e_true = 3.34 * u.TeV
        # Check for correct normalization
        test_pdf = self.edisp.data.evaluate(e_true=test_e_true)
        assert_allclose(np.sum(test_pdf), 1, atol=1e-2)
        # Check bias
        assert_allclose(self.edisp.get_bias(test_e_true), 0, atol=1e-2)
        # Check resolution
        assert_allclose(self.edisp.get_resolution(test_e_true),
                        self.resolution,
                        atol=1e-4)

    def test_io(self, tmpdir):
        indices = np.array([[1, 3, 6], [3, 3, 2]])
        desired = self.edisp.pdf_matrix[indices]
        writename = str(tmpdir / 'rmf_test.fits')
        self.edisp.write(writename)
        edisp2 = EnergyDispersion.read(writename)
        actual = edisp2.pdf_matrix[indices]
        assert_allclose(actual, desired)

    def test_apply(self):
        counts = np.arange(len(self.e_true) - 1)
        actual = self.edisp.apply(counts)
        assert_allclose(actual[0], 3.9877484855864265)

        actual = self.edisp.apply(counts, e_true=self.e_true)
        assert_allclose(actual[0], 3.9877484855864265)

    @requires_dependency('matplotlib')
    def test_plot_matrix(self):
        self.edisp.plot_matrix()

    @requires_dependency('matplotlib')
    def test_plot_bias(self):
        self.edisp.plot_bias()


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestEnergyDispersion2D:
    def setup(self):
        # TODO: use from_gauss method to create know edisp
        filename = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz'
        self.edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')

    def test_evaluation(self):
        # TODO: Move to tests for NDDataArray
        # Check that nodes are evaluated correctly
        e_node = 12
        off_node = 3
        m_node = 5
        offset = self.edisp.offset.nodes[off_node]
        energy = self.edisp.e_true.nodes[e_node]
        migra = self.edisp.migra.nodes[m_node]
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
        desired = (self.edisp.e_true.nbins,
                   self.edisp.migra.nbins,
                   self.edisp.offset.nbins)
        assert_equal(actual, desired)

    def test_get_response(self):
        # Here we test get_response with an expected gaussian shape for edisp
        from scipy.special import erf

        size_true = 50
        size_mig = 1000
        size_off = 4

        etrues = np.logspace(-1., 2., size_true + 1) * u.TeV
        migras = np.linspace(0., 4., size_mig + 1)
        offsets = np.linspace(0., 2.5, size_off + 1) * u.deg

        # Resolution with energy
        sigma = 0.15 / ((etrues[:-1] / (1 * u.TeV)).value) ** 0.3
        # Bias with energy
        mu = 1.0 + 1e-3 * (etrues[:-1] - 1 * u.TeV).value

        edisp = EnergyDispersion2D.from_gauss(etrues, migras, mu, sigma, offsets)

        for i in [5, 10, 15, 20, 25, 30, 35, 40]:
            e_true = etrues[i]
            e_reco = np.array([0.25, 0.5, 1.0, 1.5, 2.0]) * e_true
            actual = edisp.get_response(offset=0.7 * u.deg, e_true=e_true, e_reco=e_reco)

            val = ((e_reco / e_true).value - mu[i]) / (np.sqrt(2) * sigma[i])
            desired = np.diff(erf(val)) * 0.5

            # We want the absolute precision to be less than 3%
            assert_allclose(actual, desired, atol=3e-2)

    def test_exporter(self):
        # Check RMF exporter
        offset = Angle(0.612, 'deg')
        e_reco = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
        e_true = EnergyBounds.equal_log_spacing(0.8, 5, 4, 'TeV')
        rmf = self.edisp.to_energy_dispersion(offset, e_true=e_true, e_reco=e_reco)
        assert_allclose(rmf.data.data[2, 3], 0.08, atol=5e-2)  # same tolerance as above
        actual = rmf.pdf_matrix[2]
        e_val = np.sqrt(e_true[2] * e_true[3])
        desired = self.edisp.get_response(offset, e_val, e_reco)
        assert_equal(actual, desired)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.edisp.peek()
