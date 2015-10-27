# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ...irf import EnergyDispersion, EnergyDispersion2D
from ...datasets import gammapy_extra
from ...spectrum.energy import EnergyBounds


@requires_data('gammapy-extra')
def test_EnergyDispersion():
    filename = gammapy_extra.filename('test_datasets/irf/hess/ogip/run_rmf60741.fits')
    edisp = EnergyDispersion.read(filename)

    # Set PDF threshold
    threshold = 1e-2
    edisp.pdf_threshold = threshold
    a = edisp.pdf_matrix > threshold
    b = edisp.pdf_matrix == 0
    c = a + b
    actual = np.sum(c)
    desired = edisp.pdf_matrix.flatten().shape[0]
    assert_equal(actual, desired)

    # lower pdf threshold
    # assert_raises did not work
    actual = 0
    threshold = 1e-3
    try:
        edisp.pdf_threshold = threshold
    except(Exception):
        actual = 1
    desired = 1
    assert_equal(actual, desired)


@requires_data('gammapy-extra')
def test_EnergyDispersion_write(tmpdir):
    filename = gammapy_extra.filename('test_datasets/irf/hess/ogip/run_rmf60741.fits')
    edisp = EnergyDispersion.read(filename)

    indices = np.array([[1, 3, 6], [3, 3, 2]])
    desired = edisp.pdf_matrix[indices]
    writename = str(tmpdir / 'rmf_test.fits')
    edisp.write(writename)
    edisp2 = EnergyDispersion.read(writename)
    actual = edisp2.pdf_matrix[indices]
    rtol = edisp2.pdf_threshold
    assert_allclose(actual, desired, rtol=rtol)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EnergyDispersion2D():
    filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz')
    edisp = EnergyDispersion2D.read(filename)

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

    # Check that values between node make sense
    # TODO: THINK ABOUT WHAT MAKES SENSE
    energy2 = edisp.energy[e_node + 1]
    upper = edisp.evaluate(offset, energy, migra)
    lower = edisp.evaluate(offset, energy2, migra)
    e_val = (energy + energy2) / 2
    actual = edisp.evaluate(offset, e_val, migra)
    # assert_equal(lower > actual and actual > upper, True)

    # Check RMF exporter
    offset = Angle([0.612], 'deg')
    e_reco = EnergyBounds.equal_log_spacing(1, 10, 5, 'TeV')
    rmf = edisp.to_energy_dispersion(offset, e_reco, e_reco)

    # TODO: fix this test
    # actual = rmf.pdf_matrix[5]
    # desired = edisp.get_response(offset, edisp.energy[5], e_reco=e_reco)
    # assert_equal(actual, desired)
