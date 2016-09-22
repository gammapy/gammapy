# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds
from .. import CountsSpectrum


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_CountsSpectrum(tmpdir):
    # create from scratch
    counts = [0, 0, 2, 5, 17, 3] * u.ct
    bins = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')

    with pytest.raises(ValueError):
        CountsSpectrum(data=counts, energy=bins)

    bins = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
    spec = CountsSpectrum(data=counts, energy=bins)

    test_e = bins[2] + 0.1 * u.TeV
    test_eval = spec.evaluate(energy=test_e, method='nearest')
    assert_allclose(test_eval, spec.data[2])

    spec.plot()
    spec.plot_hist()

    # Test I/O 
    filename = tmpdir / 'test.fits'
    spec.write(filename)
    spec2 = CountsSpectrum.read(filename)
    expected = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
    assert_quantity_allclose(spec2.energy.data, expected)
