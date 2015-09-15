# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from numpy.testing import assert_equal
from ...spectrum import Energy, EnergyBounds
from astropy.tests.helper import remote_data
from ...datasets import get_path
from astropy.io import fits

def test_Energy():
    # Explicit constructor call
    energy = Energy([1, 3, 6, 8, 12], 'TeV')
    actual = type(energy).__module__
    desired = 'gammapy.spectrum.energy'
    assert_equal(actual, desired)

    val = u.Quantity([1, 3, 6, 8, 12], 'TeV')
    actual = Energy(val, 'GeV')
    desired = Energy((1, 3, 6, 8, 12), 'TeV')
    assert_equal(actual, desired)

    # View casting
    energy = val.view(Energy)
    actual = type(energy).__module__
    desired = 'gammapy.spectrum.energy'
    assert_equal(actual, desired)

    # New from template
    energy = Energy([0, 1, 2, 3, 4, 5], 'eV')
    energy2 = energy[1:3]
    actual = energy2
    desired = Energy([1, 2], 'eV')
    assert_equal(actual, desired)

    actual = energy2.nbins
    desired = 2
    assert_equal(actual, desired)

    actual = energy2.unit
    desired = u.eV
    assert_equal(actual, desired)

    # Equal log spacing
    energy = Energy.equal_log_spacing(1 * u.GeV, 10 * u.TeV, 6)
    actual = energy[0]
    desired = Energy(1 * u.GeV, 'TeV')
    assert_equal(actual, desired)

    energy = Energy.equal_log_spacing(2, 6, 3, 'GeV')
    actual = energy.nbins
    desired = 3
    assert_equal(actual, desired)

@remote_data
def test_EnergyBounds():
    val = u.Quantity([1, 2, 3, 4, 5], 'TeV')
    actual = EnergyBounds(val, 'GeV')
    desired = EnergyBounds((1, 2, 3, 4, 5), 'TeV')
    assert_equal(actual, desired)

    # View casting
    energy = val.view(EnergyBounds)
    actual = type(energy).__module__
    desired = 'gammapy.spectrum.energy'
    assert_equal(actual, desired)

    # New from template
    energy = EnergyBounds([0, 1, 2, 3, 4, 5], 'keV')
    energy2 = energy[1:4]
    actual = energy2
    desired = EnergyBounds([1, 2, 3], 'keV')
    assert_equal(actual, desired)

    actual = energy2.nbins
    desired = 2
    assert_equal(actual, desired)

    actual = energy2.unit
    desired = u.keV
    assert_equal(actual, desired)

    # Equal log spacing
    energy = EnergyBounds.equal_log_spacing(1 * u.TeV, 10 * u.TeV, 10)
    actual = energy.nbins
    desired = 10
    assert_equal(actual, desired)

    # Log centers
    center = energy.log_centers
    actual = type(center).__module__
    desired = 'gammapy.spectrum.energy'
    assert_equal(actual, desired)

    # Upper/lower bounds
    actual = energy.upper_bounds
    desired = energy[1:]
    assert_equal(actual, desired)

    actual = energy.lower_bounds
    desired = energy[:-1]
    assert_equal(actual, desired)

    lower = [1, 3, 4, 5]
    upper = [3, 4, 5, 8]
    actual = EnergyBounds.from_lower_and_upper_bounds(lower, upper, 'TeV')
    desired = EnergyBounds([1, 3, 4, 5, 8], 'TeV')
    assert_equal(actual, desired)

    #read EBOUNDS extension
    filename = get_path("../test_datasets/irf/hess/ogip/run_rmf60741.fits",
                        location='remote')

    hdulist = fits.open(filename);
    ebounds = EnergyBounds.from_ebounds(hdulist['EBOUNDS'])
    desired = hdulist['EBOUNDS'].data['E_MAX'][-1]
    actual = ebounds[-1].value
    assert_equal(actual, desired)

    #read MATRIX extension
    ebounds = EnergyBounds.from_rmf_matrix(hdulist['MATRIX'])
    desired = hdulist['MATRIX'].data['ENERG_LO'][3]
    actual = ebounds[3].value
    assert_equal(actual, desired)
