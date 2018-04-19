# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...irf import EffectiveAreaTable2D
from ...maps import MapAxis, WcsGeom, WcsNDMap
from ...image.models import SkyGaussian2D
from ...spectrum.models import PowerLaw
from .. import (
    SkyModel, SkyModelMapEvaluator, SkyModelMapFit, make_map_exposure_true_energy,
)


@pytest.fixture(scope='session')
def sky_model():
    spatial_model = SkyGaussian2D(
        lon_0='0.2 deg',
        lat_0='0.1 deg',
        sigma='0.2 deg',
    )
    spectral_model = PowerLaw(
        index=3,
        amplitude='1e-11 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )


@pytest.fixture(scope='session')
def geom():
    axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit=u.TeV)
    return WcsGeom.create(skydir=(0, 0), binsz=0.02, width=(8, 3),
                          coordsys='GAL', axes=[axis])


@pytest.fixture(scope='session')
def exposure(geom):
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

    pointing = SkyCoord(1, 0.5, unit='deg', frame='galactic')
    livetime = 1 * u.hour
    offset_max = 3 * u.deg
    offset = Angle('2 deg')

    exposure_map = make_map_exposure_true_energy(pointing=pointing,
                                                 livetime=livetime,
                                                 aeff=aeff,
                                                 ref_geom=geom,
                                                 offset_max=offset_max)
    return exposure_map


@pytest.fixture(scope='session')
def counts(sky_model, exposure, geom):
    evaluator = SkyModelMapEvaluator(sky_model, exposure)
    npred = evaluator.compute_npred()
    return WcsNDMap(geom, npred)


@requires_dependency('scipy')
@requires_dependency('iminuit')
@requires_data('gammapy-extra')
def test_cube_fit(sky_model, counts, exposure):
    input_model = sky_model.copy()

    input_model.parameters['lon_0'].value = 0
    input_model.parameters['index'].value = 2
    input_model.parameters['lat_0'].frozen = True
    input_model.parameters['sigma'].frozen = True

    input_model.parameters.set_parameter_errors({
        'lon_0': '0.1 deg',
        'index': '0.1',
        'amplitude': '1e-12 cm-2 s-1 TeV-1',
    })

    fit = SkyModelMapFit(model=input_model, counts=counts, exposure=exposure)
    fit.fit()

    assert_quantity_allclose(fit.model.parameters['index'].quantity,
                             sky_model.parameters['index'].quantity,
                             rtol=1e-2)
    assert_quantity_allclose(fit.model.parameters['amplitude'].quantity,
                             sky_model.parameters['amplitude'].quantity,
                             rtol=1e-2)
    assert_quantity_allclose(fit.model.parameters['lon_0'].quantity,
                             sky_model.parameters['lon_0'].quantity,
                             rtol=1e-2)

    # TODO: this gives stat = NaN. Why?
    # stat = np.sum(fit.compute_stat(), dtype='float64')
    # assert_allclose(stat, 42)
