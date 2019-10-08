# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.cube import MapDataset, PSFKernel, make_map_exposure_true_energy
from gammapy.data import GTI
from gammapy.irf import (
    EffectiveAreaTable2D,
    EnergyDependentMultiGaussPSF,
    EnergyDispersion,
)
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    BackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture
def geom():
    ebounds = np.logspace(-1.0, 1.0, 3)
    axis = MapAxis.from_edges(ebounds, name="energy", unit=u.TeV, interp="log")
    return WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.02,
        width=(2, 2),
        coordsys="CEL",
        axes=[axis],
    )


@pytest.fixture
def geom_etrue():
    ebounds_true = np.logspace(-1.0, 1.0, 4)
    axis = MapAxis.from_edges(ebounds_true, name="energy", unit=u.TeV, interp="log")
    return WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.02,
        width=(2, 2),
        coordsys="CEL",
        axes=[axis],
    )


@pytest.fixture
def geom_image():
    ebounds_true = np.logspace(-1.0, 1.0, 2)
    axis = MapAxis.from_edges(ebounds_true, name="energy", unit=u.TeV, interp="log")
    return WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis]
    )


def get_exposure(geom_etrue):
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


def get_psf(geom_etrue):
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)
    psf_kernel = PSFKernel.from_table_psf(table_psf, geom_etrue, max_radius=0.5 * u.deg)
    return psf_kernel


@pytest.fixture
def sky_model():
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


def get_map_dataset(sky_model, geom, geom_etrue, edisp=True, **kwargs):
    """This computes the total npred"""
    # define background model
    m = Map.from_geom(geom)
    m.quantity = 0.2 * np.ones(m.data.shape)
    background_model = BackgroundModel(m)

    psf = get_psf(geom_etrue)
    exposure = get_exposure(geom_etrue)

    if edisp:
        # define energy dispersion
        e_true = geom_etrue.get_axis_by_name("energy").edges
        e_reco = geom.get_axis_by_name("energy").edges
        edisp = EnergyDispersion.from_diagonal_response(e_true=e_true, e_reco=e_reco)
    else:
        edisp = None

    # define fit mask
    center = sky_model.spatial_model.position
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    mask_fit = background_model.map.geom.region_mask([circle])

    return MapDataset(
        model=sky_model,
        exposure=exposure,
        background_model=background_model,
        psf=psf,
        edisp=edisp,
        mask_fit=mask_fit,
        **kwargs
    )


@requires_data()
def test_map_dataset_str(sky_model, geom, geom_etrue):
    dataset = get_map_dataset(sky_model, geom, geom_etrue)
    dataset.counts = dataset.npred()
    dataset.mask_safe = dataset.mask_fit
    assert "MapDataset" in str(dataset)


@requires_data()
def test_fake(sky_model, geom, geom_etrue):
    """Test the fake dataset"""
    dataset = get_map_dataset(sky_model, geom, geom_etrue)
    npred = dataset.npred()
    assert np.all(npred.data >= 0)  # npred must be positive
    dataset.counts = npred
    real_dataset = dataset.copy()
    dataset.fake(314)

    assert real_dataset.counts.data.shape == dataset.counts.data.shape
    assert_allclose(real_dataset.counts.data.sum(), 8220.399727)
    assert_allclose(dataset.counts.data.sum(), 8397)


@requires_data()
def test_different_exposure_unit(sky_model, geom):
    dataset_ref = get_map_dataset(sky_model, geom, geom, edisp=False)
    npred_ref = dataset_ref.npred()

    ebounds_true = np.logspace(2, 4, 3)
    axis = MapAxis.from_edges(ebounds_true, name="energy", unit="GeV", interp="log")
    geom_gev = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.02,
        width=(2, 2),
        coordsys="CEL",
        axes=[axis],
    )

    dataset = get_map_dataset(sky_model, geom, geom_gev, edisp=False)
    npred = dataset.npred()

    assert_allclose(npred.data[0, 50, 50], npred_ref.data[0, 50, 50])


@requires_data()
def test_to_spectrum_dataset(sky_model, geom, geom_etrue):
    dataset_ref = get_map_dataset(sky_model, geom, geom_etrue, edisp=True)
    dataset_ref.counts = dataset_ref.background_model.map * 0.0
    dataset_ref.counts.data[1, 50, 50] = 1
    dataset_ref.counts.data[1, 60, 50] = 1

    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset_ref.gti = gti
    on_region = CircleSkyRegion(center=geom.center_skydir, radius=0.1 * u.deg)
    spectrum_dataset = dataset_ref.to_spectrum_dataset(on_region)

    assert np.sum(spectrum_dataset.counts.data) == 1
    assert spectrum_dataset.data_shape == (2,)
    assert spectrum_dataset.background.energy.nbin == 2
    assert spectrum_dataset.aeff.energy.nbin == 3
    assert spectrum_dataset.edisp.e_reco.nbin == 2
    assert spectrum_dataset.edisp.e_true.nbin == 3


@requires_data()
def test_map_dataset_fits_io(tmpdir, sky_model, geom, geom_etrue):
    dataset = get_map_dataset(sky_model, geom, geom_etrue)
    dataset.counts = dataset.npred()
    dataset.mask_safe = dataset.mask_fit
    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset.gti = gti

    hdulist = dataset.to_hdulist()
    actual = [hdu.name for hdu in hdulist]

    desired = [
        "PRIMARY",
        "COUNTS",
        "COUNTS_BANDS",
        "EXPOSURE",
        "EXPOSURE_BANDS",
        "BACKGROUND",
        "BACKGROUND_BANDS",
        "EDISP_MATRIX",
        "EDISP_MATRIX_EBOUNDS",
        "PSF_KERNEL",
        "PSF_KERNEL_BANDS",
        "MASK_SAFE",
        "MASK_SAFE_BANDS",
        "MASK_FIT",
        "MASK_FIT_BANDS",
        "GTI",
    ]

    assert actual == desired

    dataset.write(tmpdir / "test.fits")

    dataset_new = MapDataset.read(tmpdir / "test.fits")
    assert dataset_new.model is None
    assert dataset_new.mask.dtype == bool

    assert_allclose(dataset.counts.data, dataset_new.counts.data)
    assert_allclose(
        dataset.background_model.map.data, dataset_new.background_model.map.data
    )
    assert_allclose(dataset.edisp.data.data.value, dataset_new.edisp.data.data.value)
    assert_allclose(dataset.psf.data, dataset_new.psf.data)
    assert_allclose(dataset.exposure.data, dataset_new.exposure.data)
    assert_allclose(dataset.mask_fit, dataset_new.mask_fit)
    assert_allclose(dataset.mask_safe, dataset_new.mask_safe)

    assert dataset.counts.geom == dataset_new.counts.geom
    assert dataset.exposure.geom == dataset_new.exposure.geom
    assert dataset.background_model.map.geom == dataset_new.background_model.map.geom

    assert_allclose(
        dataset.edisp.e_true.edges.value, dataset_new.edisp.e_true.edges.value
    )
    assert dataset.edisp.e_true.unit == dataset_new.edisp.e_true.unit

    assert_allclose(
        dataset.edisp.e_reco.edges.value, dataset_new.edisp.e_reco.edges.value
    )
    assert dataset.edisp.e_true.unit == dataset_new.edisp.e_true.unit
    assert_allclose(
        dataset.gti.time_sum.to_value("s"), dataset_new.gti.time_sum.to_value("s")
    )


@requires_dependency("iminuit")
@requires_dependency("matplotlib")
@requires_data()
def test_map_fit(sky_model, geom, geom_etrue):
    dataset_1 = get_map_dataset(sky_model, geom, geom_etrue, evaluation_mode="local")
    dataset_1.background_model.norm.value = 0.5
    dataset_1.counts = dataset_1.npred()

    dataset_2 = get_map_dataset(
        sky_model, geom, geom_etrue, evaluation_mode="global", likelihood="cstat"
    )
    dataset_2.counts = dataset_2.npred()

    sky_model.parameters["sigma"].frozen = True

    dataset_1.background_model.norm.value = 0.49
    dataset_2.background_model.norm.value = 0.99

    fit = Fit([dataset_1, dataset_2])
    result = fit.run()

    assert result.success
    assert "minuit" in repr(result)

    npred = dataset_1.npred().data.sum()
    assert_allclose(npred, 6220.529956, rtol=1e-3)
    assert_allclose(result.total_stat, 11802.750562, rtol=1e-3)

    pars = result.parameters
    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("lon_0"), 0.002651, rtol=1e-2)

    assert_allclose(pars["index"].value, 3, rtol=1e-2)
    assert_allclose(pars.error("index"), 0.021222, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error("amplitude"), 3.117271e-13, rtol=1e-2)

    # background norm 1
    assert_allclose(pars[8].value, 0.5, rtol=1e-2)
    assert_allclose(pars.error(pars[8]), 0.015759, rtol=1e-2)

    # background norm 2
    assert_allclose(pars[11].value, 1, rtol=1e-2)
    assert_allclose(pars.error(pars[11]), 0.02147, rtol=1e-2)

    # test mask_safe evaluation
    mask_safe = geom.energy_mask(emin=1 * u.TeV)
    dataset_1.mask_safe = mask_safe
    dataset_2.mask_safe = mask_safe

    stat = fit.datasets.likelihood()
    assert_allclose(stat, 6425.389198)

    # test model evaluation outside image

    with pytest.raises(ValueError):
        dataset_1.model.skymodels[0].spatial_model.lon_0.value = 150
        dataset_1.npred()

    with mpl_plot_check():
        dataset_1.plot_residuals()


@requires_dependency("iminuit")
@requires_data()
def test_map_fit_one_energy_bin(sky_model, geom_image):
    dataset = get_map_dataset(sky_model, geom_image, geom_image)
    sky_model.spectral_model.index.value = 3.0
    sky_model.spectral_model.index.frozen = True
    dataset.background_model.norm.value = 0.5

    dataset.counts = dataset.npred()

    # Move a bit away from the best-fit point, to make sure the optimiser runs
    sky_model.parameters["sigma"].value = 0.21
    dataset.background_model.parameters["norm"].frozen = True

    fit = Fit(dataset)
    result = fit.run()

    assert result.success

    npred = dataset.npred().data.sum()
    assert_allclose(npred, 4076.779039, rtol=1e-3)
    assert_allclose(result.total_stat, 5722.439112, rtol=1e-3)

    pars = result.parameters

    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("lon_0"), 0.00407, rtol=1e-2)

    assert_allclose(pars["sigma"].value, 0.2, rtol=1e-2)
    assert_allclose(pars.error("sigma"), 0.00237, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars.error("amplitude"), 1.901406e-13, rtol=1e-2)


def test_create(geom, geom_etrue):
    # tests empty datasets created

    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")
    rad_axis = MapAxis(nodes=np.linspace(0.0, 1.0, 51), unit="deg", name="theta")

    empty_dataset = MapDataset.create(geom, geom_etrue, migra_axis, rad_axis)

    assert_allclose(empty_dataset.counts.data.sum(), 0.0)
    assert_allclose(empty_dataset.background_model.map.data.sum(), 0.0)

    assert empty_dataset.psf.psf_map.data.shape == (3, 50, 100, 100)
    assert empty_dataset.psf.exposure_map.data.shape == (3, 1, 100, 100)

    assert empty_dataset.edisp.edisp_map.data.shape == (3, 50, 100, 100)
    assert empty_dataset.edisp.exposure_map.data.shape == (3, 1, 100, 100)

    assert_allclose(empty_dataset.edisp.edisp_map.data.sum(), 30000)

    assert_allclose(empty_dataset.gti.time_delta, 0.0 * u.s)


@requires_data()
def test_stack(geom, geom_etrue):

    m = Map.from_geom(geom)
    m.quantity = 0.2 * np.ones(m.data.shape)
    background_model1 = BackgroundModel(m)
    c_map1 = Map.from_geom(geom)
    c_map1.quantity = 0.3 * np.ones(c_map1.data.shape)
    mask1 = np.ones(m.data.shape, dtype=bool)
    mask1[0][0][0:10] = False

    dataset1 = MapDataset(
        counts=c_map1,
        background_model=background_model1,
        exposure=get_exposure(geom_etrue),
        mask_safe=mask1,
    )

    c_map2 = Map.from_geom(geom)
    c_map2.quantity = 0.1 * np.ones(c_map2.data.shape)
    background_model2 = BackgroundModel(m, norm=0.5)
    mask2 = np.ones(m.data.shape, dtype=bool)
    mask2[0][3] = False

    dataset2 = MapDataset(
        counts=c_map2,
        background_model=background_model2,
        exposure=get_exposure(geom_etrue),
        mask_safe=mask2,
    )
    dataset1.stack(dataset2)
    assert_allclose(dataset1.counts.data.sum(), 7987)
    assert_allclose(dataset1.background_model.map.data.sum(), 5988)
    assert_allclose(dataset1.exposure.data, 2.0 * dataset2.exposure.data)
    assert_allclose(dataset1.mask_safe.sum(), 20000)
