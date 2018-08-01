# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...maps import WcsGeom, HpxGeom, MapAxis
from ...irf import EffectiveAreaTable2D
from ..exposure import make_map_exposure_true_energy

pytest.importorskip('scipy')
pytest.importorskip('healpy')


@pytest.fixture(scope='session')
def aeff():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    return EffectiveAreaTable2D.read(filename, hdu='AEFF_2D')


def geom(map_type, ebounds):
    axis = MapAxis.from_edges(ebounds, name="energy", unit='TeV', interp='log')
    if map_type == 'wcs':
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])
    elif map_type == 'hpx':
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data('gammapy-extra')
@pytest.mark.parametrize("pars", [
    {
        'geom': geom(map_type='wcs', ebounds=[0.1, 1, 10]),
        'shape': (2, 3, 4),
        'sum': 54448477.348027,
    },
    {
        'geom': geom(map_type='wcs', ebounds=[0.1, 10]),
        'shape': (1, 3, 4),
        'sum': 31219048.597406,
    },
    # TODO: make this work for HPX
    # 'HpxGeom' object has no attribute 'separation'
    # {
    #     'geom': geom(map_type='hpx', ebounds=[0.1, 1, 10]),
    #     'shape': '???',
    #     'sum': '???',
    # },
])
def test_make_map_exposure_true_energy(aeff, pars):
    m = make_map_exposure_true_energy(
        pointing=SkyCoord(2, 1, unit='deg'),
        livetime='42 s', aeff=aeff, geom=pars['geom'],
    )

    assert m.data.shape == pars['shape']
    assert m.unit == 'm2 s'
    assert_allclose(m.data.sum(), pars['sum'])
