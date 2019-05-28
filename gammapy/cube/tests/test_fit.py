# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency
from ...utils.fitting import Fit
from ...irf import EffectiveAreaTable2D, EnergyDependentMultiGaussPSF
from ...irf.energy_dispersion import EnergyDispersion
from ...maps import MapAxis, WcsGeom, WcsNDMap, Map
from ...image.models import SkyGaussian
from ...spectrum.models import PowerLaw
from ..models import SkyModel, BackgroundModel
from .. import MapDataset, make_map_exposure_true_energy, PSFKernel


def geom(ebounds):
    axis = MapAxis.from_edges(ebounds, name="energy", unit=u.TeV)
    return WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis]
    )


def geom_etrue(ebounds_true):
    axis = MapAxis.from_edges(ebounds_true, name="energy", unit=u.TeV)
    return WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis]
    )


def exposure(geom_etrue):
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

    exposure_map = make_map_exposure_true_energy(
        pointing=SkyCoord(1, 0.5, unit="deg", frame="galactic"),
        livetime="1 hour",
        aeff=aeff,
        geom=geom_etrue,
    )
    return exposure_map


def background(geom):
    m = Map.from_geom(geom)
    m.quantity = 0.2 * np.ones(m.data.shape)
    return m


def edisp(geom, geom_etrue):
    e_true = geom_etrue.get_axis_by_name("energy").edges
    e_reco = geom.get_axis_by_name("energy").edges
    return EnergyDispersion.from_diagonal_response(e_true=e_true, e_reco=e_reco)


def psf(geom_etrue):
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)
    psf_kernel = PSFKernel.from_table_psf(table_psf, geom_etrue, max_radius=0.5 * u.deg)
    return psf_kernel


@pytest.fixture
def sky_model():
    spatial_model = SkyGaussian(lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg")
    spectral_model = PowerLaw(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


def mask_fit(geom, sky_model):
    p = sky_model.spatial_model.parameters
    center = SkyCoord(p["lon_0"].value, p["lat_0"].value, frame="galactic", unit="deg")
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    return geom.region_mask([circle])


def counts(sky_model, exposure, background, psf, edisp):
    """This computes the total npred"""
    npred = MapDataset(
        model=sky_model,
        exposure=exposure,
        background_model=background,
        psf=psf,
        edisp=edisp,
    ).npred()
    return npred


@requires_dependency("iminuit")
@requires_data()
def test_map_fit(sky_model):
    ebounds = np.logspace(-1.0, 1.0, 3)
    ebounds_true = np.logspace(-1.0, 1.0, 4)
    geom_r = geom(ebounds)
    geom_t = geom_etrue(ebounds_true)

    background_map = background(geom_r)
    background_model_1 = BackgroundModel(background_map, norm=0.5)
    background_model_2 = BackgroundModel(background_map, norm=1)

    psf_map = psf(geom_t)
    edisp_map = edisp(geom_r, geom_t)
    exposure_map = exposure(geom_t)
    counts_map_1 = counts(
        sky_model, exposure_map, background_model_1, psf_map, edisp_map
    )
    counts_map_2 = counts(
        sky_model, exposure_map, background_model_2, psf_map, edisp_map
    )

    mask_map = mask_fit(geom_r, sky_model)
    sky_model.parameters["sigma"].frozen = True

    dataset_1 = MapDataset(
        model=sky_model,
        counts=counts_map_1,
        exposure=exposure_map,
        mask_fit=mask_map,
        psf=psf_map,
        edisp=edisp_map,
        background_model=background_model_1,
        evaluation_mode="local",
    )

    dataset_2 = MapDataset(
        model=sky_model,
        counts=counts_map_2,
        exposure=exposure_map,
        mask_fit=mask_map,
        psf=psf_map,
        edisp=edisp_map,
        background_model=background_model_2,
        evaluation_mode="global",
        likelihood="cstat",
    )

    background_model_1.parameters["norm"].value = 0.4
    background_model_2.parameters["norm"].value = 0.9

    fit = Fit([dataset_1, dataset_2])
    result = fit.run()

    assert result.success
    assert "minuit" in repr(result)

    npred = dataset_1.npred().data.sum()
    assert_allclose(npred, 4454.932873, rtol=1e-3)
    assert_allclose(result.total_stat, 12728.351643, rtol=1e-3)

    pars = result.parameters
    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("lon_0"), 0.003627, rtol=1e-2)

    assert_allclose(pars["index"].value, 3, rtol=1e-2)
    assert_allclose(pars.error("index"), 0.031294, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error("amplitude"), 3.885326e-13, rtol=1e-2)

    # background norm 1
    assert_allclose(pars[6].value, 0.5, rtol=1e-2)
    assert_allclose(pars.error(pars[6]), 0.015399, rtol=1e-2)

    # background norm 2
    assert_allclose(pars[9].value, 1, rtol=1e-2)
    assert_allclose(pars.error(pars[9]), 0.02104, rtol=1e-2)

    # test mask_safe evaluation
    mask_safe = geom_r.energy_mask(emin=1 * u.TeV)
    dataset_1.mask_safe = mask_safe
    dataset_2.mask_safe = mask_safe

    stat = fit.datasets.likelihood()
    assert_allclose(stat, 5895.205587)

    # test model evaluation outside image

    with pytest.raises(ValueError):
        dataset_1.model.skymodels[0].spatial_model.lon_0.value = 150
        dataset_1.npred()


@requires_dependency("iminuit")
@requires_data()
def test_map_fit_one_energy_bin(sky_model):
    ebounds = np.logspace(-1.0, 1.0, 2)
    geom_r = geom(ebounds)

    background_map = background(geom_r)
    background_model = BackgroundModel(background_map, norm=0.5, tilt=0.0)
    psf_map = psf(geom_r)
    edisp_map = edisp(geom_r, geom_r)
    exposure_map = exposure(geom_r)
    counts_map = counts(sky_model, exposure_map, background_model, psf_map, edisp_map)
    mask_map = mask_fit(geom_r, sky_model)

    sky_model.parameters["index"].value = 3.0
    sky_model.parameters["index"].frozen = True
    # Move a bit away from the best-fit point, to make sure the optimiser runs
    sky_model.parameters["sigma"].value = 0.21
    background_model.parameters["norm"].frozen = True

    dataset = MapDataset(
        model=sky_model,
        counts=counts_map,
        exposure=exposure_map,
        mask_fit=mask_map,
        psf=psf_map,
        edisp=edisp_map,
        background_model=background_model,
    )
    fit = Fit(dataset)
    result = fit.run()

    assert result.success

    npred = dataset.npred().data.sum()
    assert_allclose(npred, 1087.073518, rtol=1e-3)
    assert_allclose(result.total_stat, 5177.19198, rtol=1e-3)

    pars = result.parameters

    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("lon_0"), 0.04623, rtol=1e-2)

    assert_allclose(pars["sigma"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("sigma"), 0.031759, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error("amplitude"), 2.163318e-12, rtol=1e-2)
