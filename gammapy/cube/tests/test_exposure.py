from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...irf import EffectiveAreaTable2D, Background3D
from .. import make_exposure_cube, make_background_cube
from .. import SkyCube


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@pytest.fixture(scope='session')
def aeff():
    aeff_filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    return EffectiveAreaTable2D.read(aeff_filename, hdu='AEFF_2D')


@pytest.fixture(scope='session')
def counts_cube():
    ccube_filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits'
    return SkyCube.read(ccube_filename, format='fermi-counts')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_exposure_cube(aeff, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    offset_max = Angle(2.2, 'deg')

    exp_cube = make_exposure_cube(
        pointing, livetime, aeff, counts_cube, offset_max=offset_max,
    )
    exp_ref = Quantity(4.7e8, 'm2 s')
    coordinates = exp_cube.sky_image_ref.coordinates()
    offset = coordinates.separation(pointing)

    assert np.shape(exp_cube.data)[1:] == np.shape(counts_cube.data)[1:]
    assert np.shape(exp_cube.data)[0] == np.shape(counts_cube.data)[0]
    assert exp_cube.wcs == counts_cube.wcs
    assert_quantity_allclose(np.nanmax(exp_cube.data), exp_ref, rtol=100)
    assert exp_cube.data.unit == exp_ref.unit
    assert exp_cube.data[:, offset > offset_max].sum() == 0


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_cube(bkg_3d, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    offset_max = Angle(2.2, 'deg')

    bkg_cube = make_background_cube(
        pointing, livetime, bkg_3d, counts_cube, offset_max=offset_max,
    )

    assert bkg_cube.data.shape == (15, 120, 200)
    assert bkg_cube.data.unit == ''

    print(bkg_cube.data.sum())
    assert_allclose(bkg_cube.data[0, 0, 0], 0.013959329891790048)
    assert_allclose(bkg_cube.data.sum(), 1315.7910319477235)

    # Check that `offset_max` is working properly
    pos = SkyCoord(85.6, 23, unit='deg')
    val = bkg_cube.lookup(pos, energy=1 * u.TeV)
    assert_allclose(val, 0)
