# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency
from ...irf import EffectiveAreaTable2D, EnergyDependentMultiGaussPSF
from ...irf.energy_dispersion import EnergyDispersion
from ...maps import MapAxis, WcsGeom, WcsNDMap, Map
from ...image.models import SkyGaussian
from ...spectrum.models import PowerLaw
from ..models import SkyModel
from .. import MapEvaluator, MapFit, make_map_exposure_true_energy, PSFKernel


@pytest.fixture(scope="session")
def geom():
    axis = MapAxis.from_edges(np.logspace(-1., 1., 3), name="energy", unit=u.TeV)
    return WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis]
    )


@pytest.fixture(scope="session")
def exposure(geom):
    filename = "$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

    exposure_map = make_map_exposure_true_energy(
        pointing=SkyCoord(1, 0.5, unit="deg", frame="galactic"),
        livetime="1 hour",
        aeff=aeff,
        geom=geom,
    )
    return exposure_map


@pytest.fixture(scope="session")
def background(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones(m.data.shape) * 1e-5
    return m


@pytest.fixture(scope="session")
def edisp(geom):
    e_true = geom.get_axis_by_name("energy").edges
    return EnergyDispersion.from_diagonal_response(e_true=e_true)


@pytest.fixture(scope="session")
def psf(geom):
    filename = "$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)
    psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius=0.5 * u.deg)
    return psf_kernel


@pytest.fixture
def sky_model():
    spatial_model = SkyGaussian(lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg")
    spectral_model = PowerLaw(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


@pytest.fixture
def mask(geom, sky_model):
    p = sky_model.spatial_model.parameters
    center = SkyCoord(p["lon_0"].value, p["lat_0"].value, frame="galactic", unit="deg")
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    data = geom.region_mask([circle])
    return WcsNDMap(geom=geom, data=data)


@pytest.fixture
def counts(sky_model, exposure, background, psf, edisp):
    evaluator = MapEvaluator(
        model=sky_model, exposure=exposure, background=background, psf=psf, edisp=edisp
    )
    npred = evaluator.compute_npred()
    return WcsNDMap(exposure.geom, npred)


@requires_dependency("scipy")
@requires_dependency("iminuit")
@requires_data("gammapy-extra")
def test_cube_fit(sky_model, counts, exposure, psf, background, mask, edisp):
    sky_model.parameters["lon_0"].value = 0.5
    sky_model.parameters["lat_0"].value = 0.5
    sky_model.parameters["index"].value = 2
    sky_model.parameters["sigma"].frozen = True

    fit = MapFit(
        model=sky_model,
        counts=counts,
        exposure=exposure,
        background=background,
        mask=mask,
        psf=psf,
        edisp=edisp,
    )
    result = fit.run()

    assert sky_model is not fit._model
    assert sky_model is not result.model
    assert result.success
    assert "minuit" in repr(result)

    stat_expected = 3840.0605649268496
    assert_allclose(result.total_stat, stat_expected, rtol=1e-2)

    pars = result.model.parameters
    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("lon_0"), 0.005895, rtol=1e-2)

    assert_allclose(pars["index"].value, 3, rtol=1e-2)
    assert_allclose(pars.error("index"), 0.05614, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error("amplitude"), 3.936e-13, rtol=1e-2)

    assert result.model.spectral_model.parameters.covariance is not None
    assert result.model.spatial_model.parameters.covariance is not None
