# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...maps import WcsGeom, HpxGeom, MapAxis, WcsNDMap
from ...irf import EffectiveAreaTable2D
from ..exposure import make_map_exposure_true_energy, weighted_exposure_image
from ...spectrum.models import ConstantModel
pytest.importorskip('scipy')
pytest.importorskip('healpy')


@pytest.fixture(scope='session')
def aeff():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc//caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    return EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')


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
        'sum': 8.103974e+08,
    },
    {
        'geom': geom(map_type='wcs', ebounds=[0.1, 10]),
        'shape': (1, 3, 4),
        'sum': 2.387916e+08,
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
        livetime='42 s',
        aeff=aeff,
        geom=pars['geom'],
    )

    assert m.data.shape == pars['shape']
    assert m.unit == 'm2 s'
    assert_allclose(m.data.sum(), pars['sum'], rtol=1e-5)

def test_weighted_exposure_image():
    # Create fake exposure Map
    expo_map = WcsNDMap.create(npix=10, binsz=1., axes=[MapAxis(np.logspace(-1.,1.,11), unit='TeV', name='energy')],
                        unit='m2s')
    expo_map.data += 1.

    cst_model = ConstantModel(2.)
    weighted_expo = weighted_exposure_image(expo_map,cst_model)
    assert_allclose(weighted_expo.data.sum(), 100)