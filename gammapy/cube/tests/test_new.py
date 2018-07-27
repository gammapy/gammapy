# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data
from ...irf import EffectiveAreaTable2D, Background3D
from ...maps import WcsNDMap, WcsGeom, MapAxis
from ..new import make_map_separation, make_map_exposure_true_energy, make_map_background_irf
from ..mapmaker import MapMaker
from ...data import DataStore

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@pytest.fixture(scope='session')
def aeff():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    return EffectiveAreaTable2D.read(filename, hdu='AEFF_2D')


@pytest.fixture(scope='session')
def counts_cube():
    import os
    filename = os.path.join(
        os.environ['GAMMAPY_EXTRA'],
        'datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits'
    )
    return WcsNDMap.read(filename)


def test_separation_map():
    geom = WcsGeom.create(skydir=(0, 0), npix=10,
                          binsz=0.1, coordsys='GAL', proj='CAR',
                          axes=[MapAxis.from_edges([0, 2, 3])])
    position = SkyCoord(1, 0, unit='deg', frame='galactic').icrs
    m = make_map_separation(geom, position)

    assert m.unit == 'deg'
    assert m.data.shape == (10, 10)
    assert_allclose(m.data[0, 0], 0.7106291438079875)

    # Make sure it also works for 2D maps as input
    geom = m.geom.to_image()
    m = make_map_separation(geom, position)
    assert m.unit == 'deg'
    assert m.data.shape == (10, 10)
    assert_allclose(m.data[0, 0], 0.7106291438079875)


@requires_data('gammapy-extra')
def test_make_map_exposure_true_energy(aeff, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    offset_max = Angle(2.2, 'deg')

    m = make_map_exposure_true_energy(
        pointing, livetime, aeff, counts_cube.geom, offset_max,
    )

    assert m.data.shape == (15, 120, 200)
    assert m.unit == 'm2 s'
    assert_quantity_allclose(np.nanmax(m.data), 4.7e8, rtol=100)


@requires_data('gammapy-extra')
def test_make_map_fov_background(bkg_3d, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    offset_max = Angle(2.2, 'deg')

    m = make_map_background_irf(
        pointing, livetime, bkg_3d, counts_cube.geom, offset_max,
    )

    assert m.data.shape == (15, 120, 200)
    assert_allclose(m.data[0, 0, 0], 0.013759879207779322)
    assert_allclose(m.data.sum(), 1301.0242859188463)

    # TODO: Check that `offset_max` is working properly
    # pos = SkyCoord(85.6, 23, unit='deg')
    # val = bkg_cube.lookup(pos, energy=1 * u.TeV)
    # assert_allclose(val, 0)


@requires_data('gammapy-extra')
@pytest.mark.parametrize("mode, expected", [("trim", 107214.0), ("strict", 53486.0)])
def test_MapMaker(mode,expected):
    ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/")
    pos_SagA = SkyCoord(266.41681663, -29.00782497, unit="deg", frame="icrs")
    energy_axis = MapAxis.from_edges([0.1,0.5,1.5,3.0,10.],name='energy',unit='TeV',interp='log')
    geom = WcsGeom.create(binsz=0.1*u.deg, skydir=pos_SagA, width=15.0, axes=[energy_axis])
    mmaker = MapMaker(geom, geom, offset_max=6.0 * u.deg, cutout_mode=mode)
    obs = [110380, 111140]

    for obsid in obs:
        mmaker.process_obs(ds.obs(obsid))
    assert mmaker.exposure_map.unit == "m2 s"
    assert_quantity_allclose(mmaker.counts_map.data.sum(), expected)


    etrue_axis = MapAxis.from_bounds(0.5, 50.0, 10, name='energy', unit='TeV', interp='log')
    geom_etrue = WcsGeom.create(binsz=0.1*u.deg, skydir=pos_SagA, width=15.0, axes=[etrue_axis])
    maker = MapMaker(geom, geom_etrue, offset_max=6.0 * u.deg, cutout_mode=mode)
    obslist = ds.obs_list(obs)
    maps = maker.run(obslist)
    assert maps['exposure_map'].unit == "m2 s"
    assert_quantity_allclose(maps['counts_map'].data.sum(), expected)


