# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...irf import EnergyDispersion, EnergyDispersion2D
from ...datasets import gammapy_extra
from ...utils.energy import EnergyBounds


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
        assert_allclose(np.sum(test_pdf), 1, atol=1e-4)
        # Check bias
        assert_allclose(self.edisp.get_bias(test_e_true), 0, atol=1e-4)
        # Check resolution
        assert_allclose(self.edisp.get_resolution(test_e_true),
                        self.resolution,
                        atol=1e-4)

    requires_dependency('matplotlib')
    def check_plot(self):
        edisp.plot_matrix()
        edisp.plot_bias()

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

@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestEnergyDispersion2D():
    def setup(self):
        filename = gammapy_extra.filename(
            'test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz')
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
        energy = [1,2] * u.TeV
        migra = [0.98, 0.97, 0.7]
        offset = [0.1, 0.2] * u.deg
        actual = self.edisp.data.evaluate(e_true=energy, migra=migra, offset=offset)
        assert_allclose(actual.shape, (2,3,2))

        # Check evaluation at all nodes
        actual = self.edisp.data.evaluate().shape
        desired = (self.edisp.e_true.nbins,
                   self.edisp.migra.nbins,
                   self.edisp.offset.nbins)
        assert_equal(actual, desired)

    def test_get_response(self):
        # Get response
        e_reco = EnergyBounds.equal_log_spacing(1 * u.GeV, 100 * u.TeV, 100)
        response = self.edisp.get_response(1 * u.deg, 1 * u.TeV, e_reco)
        actual = len(response)
        desired = e_reco.nbins
        assert_equal(actual, desired)

        # Check normalization
        actual = np.sum(response)
        desired = 1
        assert_allclose(actual, desired, rtol=1e-1)

        # Check value
        offset = 0.2 * u.deg
        e_true = 1.2 * u.TeV
        e_reco = EnergyBounds.from_lower_and_upper_bounds(
            self.edisp.migra.lo * e_true, self.edisp.migra.hi * e_true)
        response2 = self.edisp.get_response(offset=offset,
                                            e_true=e_true,
                                            e_reco=e_reco)
        assert_allclose(response2[20], 9.2941217306683827e-05) 

    def test_exporter(self):
        # Check RMF exporter
        offset = Angle(0.612, 'deg')
        e_reco = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
        e_true = EnergyBounds.equal_log_spacing(0.8, 5, 4, 'TeV')
        rmf = self.edisp.to_energy_dispersion(offset, e_true=e_true, e_reco=e_reco)
        assert_allclose(rmf.data.data[2,3], 0.10531216786)
        actual = rmf.pdf_matrix[2]
        e_val = np.sqrt(e_true[2] * e_true[3])
        desired = self.edisp.get_response(offset, e_val, e_reco)
        assert_equal(actual, desired)

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.edisp.peek()
