# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/test_datasets/cta_1dc/caldb/data/cta/prod3b/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_basics(bkg_3d):
    assert 'NDDataArray summary info' in str(bkg_3d.data)

    axis = bkg_3d.data.axis('energy')
    assert axis.nbins == 21
    assert axis.unit == 'TeV'

    axis = bkg_3d.data.axis('detx')
    assert axis.nbins == 12
    assert axis.unit == 'deg'

    axis = bkg_3d.data.axis('dety')
    assert axis.nbins == 12
    assert axis.unit == 'deg'

    data = bkg_3d.data.data
    assert data.shape == (21, 12, 12)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_evalutate(bkg_3d):
    bkg_rate = bkg_3d.data.evaluate(energy='1 TeV', detx='0.2 deg', dety='0.5 deg')
    assert_allclose(bkg_rate.value, 0.00013652553025167435)
    bkg_rate.unit == u.Unit('s-1 MeV-1 sr-1')
