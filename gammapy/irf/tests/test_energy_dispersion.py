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
def test_EnergyDispersion():
    e_true = np.logspace(0, 1, 101) * u.TeV
    e_reco = e_true
    resolution = 0.1
    edisp = EnergyDispersion.from_gauss(e_true=e_true, e_reco=e_reco,
                                        pdf_threshold=1e-7,
                                        sigma=resolution)

    test_e = 3.34 * u.TeV
    # Check for correct normalization
    test_pdf = edisp.evaluate(e_true=test_e)
    assert_allclose(np.sum(test_pdf), 1, atol=1e-4)
    # Check bias
    assert_allclose(edisp.get_bias(test_e), 0, atol=1e-4)
    # Check resolution
    assert_allclose(edisp.get_resolution(test_e), resolution, atol=1e-4)

    # Check plotting methods
    edisp.plot_matrix()
    edisp.plot_bias()


@requires_data('gammapy-extra')
def test_EnergyDispersion_write(tmpdir):
    filename = gammapy_extra.filename(
        'test_datasets/irf/hess/ogip/run_rmf60741.fits')
    edisp = EnergyDispersion.read(filename)

    indices = np.array([[1, 3, 6], [3, 3, 2]])
    desired = edisp.pdf_matrix[indices]
    writename = str(tmpdir / 'rmf_test.fits')
    edisp.write(writename)
    edisp2 = EnergyDispersion.read(writename)
    actual = edisp2.pdf_matrix[indices]
    assert_allclose(actual, desired)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EnergyDispersion2D():
    filename = gammapy_extra.filename(
        'test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz')
    edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')

    # Check that nodes are evaluated correctly
    e_node = 12
    off_node = 3
    m_node = 5
    offset = edisp.offset[off_node]
    energy = edisp.energy[e_node]
    migra = edisp.migra[m_node]
    actual = edisp.evaluate(offset, energy, migra)
    desired = edisp.dispersion[off_node, m_node, e_node]
    assert_allclose(actual, desired, rtol=1e-06)

    # Check evaluation at all nodes
    actual = edisp.evaluate().shape
    desired = (len(edisp.offset), len(edisp.migra), len(edisp.energy))
    assert_equal(actual, desired)

    # Get response
    e_reco = EnergyBounds.equal_log_spacing(1 * u.GeV, 100 * u.TeV, 100)
    response = edisp.get_response('1 deg', '1 TeV', e_reco)
    actual = len(response)
    desired = e_reco.nbins
    assert_equal(actual, desired)

    # Check normalization
    actual = np.sum(response)
    desired = 1
    assert_allclose(actual, desired, rtol=1e-1)

    # Check RMF exporter
    offset = Angle([0.612], 'deg')
    e_reco = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
    e_true = EnergyBounds.equal_log_spacing(0.8, 5, 4, 'TeV')
    rmf = edisp.to_energy_dispersion(offset, e_true=e_true, e_reco=e_reco)
    actual = rmf.pdf_matrix[2]
    e_val = np.sqrt(e_true[2] * e_true[3])
    desired = edisp.get_response(offset, e_val, e_reco)
    assert_equal(actual, desired)
