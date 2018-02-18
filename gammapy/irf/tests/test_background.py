# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D, Background2D
from ...utils.fits import table_to_fits_table


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_basics(bkg_3d):
    assert 'NDDataArray summary info' in str(bkg_3d.data)

    axis = bkg_3d.data.axis('energy')
    assert axis.nbins == 21
    assert axis.unit == 'TeV'

    axis = bkg_3d.data.axis('detx')
    assert axis.nbins == 36
    assert axis.unit == 'deg'

    axis = bkg_3d.data.axis('dety')
    assert axis.nbins == 36
    assert axis.unit == 'deg'

    data = bkg_3d.data.data
    assert data.shape == (21, 36, 36)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_evaluate(bkg_3d):
    bkg_rate = bkg_3d.data.evaluate(energy='1 TeV', detx='0.2 deg', dety='0.5 deg')
    assert_allclose(bkg_rate.value, 0.00013352689711418575)
    assert bkg_rate.unit == 's-1 MeV-1 sr-1'


@requires_data('gammapy-extra')
def test_background_3d_write(bkg_3d):
    hdu = table_to_fits_table(bkg_3d.to_table())
    assert_equal(hdu.data['DETX_LO'][0], bkg_3d.data.axis('detx').lo.value)
    assert hdu.header['TUNIT1'] == bkg_3d.data.axis('detx').lo.unit


@pytest.fixture(scope='session')
def bkg_2d():
    """A simple Background2D test case"""
    energy = [0.1, 10, 1000] * u.TeV
    offset = [0, 1, 2, 3] * u.deg
    data = np.zeros((2, 3)) * u.Unit('s-1 MeV-1 sr-1')
    data.value[1, 0] = 2
    data.value[1, 1] = 4
    return Background2D(
        energy_lo=energy[:-1], energy_hi=energy[1:],
        offset_lo=offset[:-1], offset_hi=offset[1:],
        data=data,
    )


@requires_dependency('scipy')
def test_background_2d_evaluate(bkg_2d):
    # TODO: the test cases here can probably be improved a bit
    # There's some redundancy, and no case exactly at a node in energy

    # Evaluate at log center between nodes in energy
    res = bkg_2d.evaluate(fov_offset=[1, 0.5] * u.deg, energy_reco=1 * u.TeV)
    assert_allclose(res.value, 0)
    assert res.shape == (2,)
    assert res.unit == 's-1 MeV-1 sr-1'

    res = bkg_2d.evaluate(fov_offset=[1, 0.5] * u.deg, energy_reco=100 * u.TeV)
    assert_allclose(res.value, [3, 2])

    res = bkg_2d.evaluate(fov_offset=[1, 0.5] * u.deg, energy_reco=[1, 100] * u.TeV)
    assert_allclose(res.value, [[0, 0], [3, 2]])
    assert res.shape == (2, 2)

    res = bkg_2d.evaluate(fov_offset=1 * u.deg, energy_reco=[1, 100] * u.TeV)
    assert_allclose(res.value, [0, 3])
    assert res.shape == (2,)


def test_background_2d_read_write(tmpdir, bkg_2d):
    filename = str(tmpdir / "bkg2d.fits")
    bkg_2d.to_fits().writeto(filename)

    bkg_2d_2 = Background2D.read(filename)

    axis = bkg_2d_2.data.axis('energy')
    assert axis.nbins == 2
    assert axis.unit == 'TeV'

    axis = bkg_2d_2.data.axis('offset')
    assert axis.nbins == 3
    assert axis.unit == 'deg'

    data = bkg_2d_2.data.data
    assert data.shape == (2, 3)
    assert data.unit == 's-1 MeV-1 sr-1'
