# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data, requires_dependency
from ...irf import EffectiveAreaTable2D, EnergyDependentMultiGaussPSF
from ...irf.energy_dispersion import EnergyDispersion
from ...maps import MapAxis, WcsGeom, WcsNDMap, Map
from ...image.models import SkyGaussian
from ...spectrum.models import PowerLaw
from .. import (
    SkyModel,
    MapEvaluator,
    MapFit,
    make_map_exposure_true_energy,
    PSFKernel,
)


@pytest.fixture(scope='session')
def geom():
    axis = MapAxis.from_edges(np.logspace(-1., 1., 3), name="energy", unit=u.TeV)
    return WcsGeom.create(skydir=(0, 0), binsz=0.02, width=(2, 2),
                          coordsys='GAL', axes=[axis])


@pytest.fixture(scope='session')
def exposure(geom):
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

    exposure_map = make_map_exposure_true_energy(
        pointing=SkyCoord(1, 0.5, unit='deg', frame='galactic'),
        livetime='1 hour',
        aeff=aeff,
        geom=geom,
    )
    return exposure_map


@pytest.fixture(scope='session')
def background(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones(m.data.shape) * 1e-5
    return m


@pytest.fixture(scope='session')
def edisp(geom):
    e_true = geom.get_axis_by_name('energy').edges
    return EnergyDispersion.from_diagonal_matrix(e_true=e_true)


@pytest.fixture(scope='session')
def psf(geom):
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')

    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)
    psf_kernel = PSFKernel.from_table_psf(table_psf,
                                          geom,
                                          max_radius=0.5 * u.deg)
    return psf_kernel


@pytest.fixture
def sky_model():
    spatial_model = SkyGaussian(
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


@pytest.fixture
def counts(sky_model, exposure, background, psf, edisp):
    evaluator = MapEvaluator(
        sky_model=sky_model,
        exposure=exposure,
        background=background,
        psf=psf,
        edisp=edisp,
    )
    npred = evaluator.compute_npred()
    return WcsNDMap(exposure.geom, npred)


@requires_dependency('scipy')
@requires_dependency('iminuit')
@requires_data('gammapy-extra')
def test_cube_fit(sky_model, counts, exposure, psf, background, edisp):
    sky_model.parameters['gaussian.lon_0'].value = 0.5
    sky_model.parameters['gaussian.lat_0'].value = 0.5
    sky_model.parameters['powerlaw.index'].value = 2
    sky_model.parameters['gaussian.sigma'].frozen = True

    sky_model.parameters.set_parameter_errors({
        'gaussian.lon_0': '0.01 deg',
        'gaussian.lat_0': '0.01 deg',
        'gaussian.sigma': '0.02 deg',
        'powerlaw.index': 0.1,
        'powerlaw.amplitude': '1e-13 cm-2 s-1 TeV-1',
    })

    fit = MapFit(
        model=sky_model,
        counts=counts,
        exposure=exposure,
        background=background,
        psf=psf,
        edisp=edisp,
    )
    fit.fit()
    pars = fit.model.parameters

    assert sky_model is fit.model
    assert sky_model.parameters['gaussian.lon_0'] is fit.model.parameters['gaussian.lon_0']
    assert sky_model.parameters['gaussian.lon_0'] is sky_model.spatial_model.parameters['gaussian.lon_0']

    assert_allclose(pars['gaussian.lon_0'].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error('gaussian.lon_0'), 0.005895, rtol=1e-2)

    assert_allclose(pars['powerlaw.index'].value, 3, rtol=1e-2)
    assert_allclose(pars.error('powerlaw.index'), 0.05614, rtol=1e-2)

    assert_allclose(pars['powerlaw.amplitude'].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error('powerlaw.amplitude'), 3.936e-13, rtol=1e-2)

    stat = np.sum(fit.stat, dtype='float64')
    stat_expected = 3840.0605649268496
    assert_allclose(stat, stat_expected, rtol=1e-2)
