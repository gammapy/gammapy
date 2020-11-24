# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.data import GTI
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff
from gammapy.irf import (
    EDispKernel,
    EDispKernelMap,
    EDispMap,
    EffectiveAreaTable2D,
    EnergyDependentMultiGaussPSF,
    PSFMap,
)
from gammapy.makers.utils import make_map_exposure_true_energy
from gammapy.maps import Map, MapAxis, WcsGeom, WcsNDMap
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture
def geom():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2)
    return WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.02,
        width=(2, 2),
        frame="icrs",
        axes=[axis],
    )


@pytest.fixture
def geom_etrue():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3, name="energy_true")
    return WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.02,
        width=(2, 2),
        frame="icrs",
        axes=[axis],
    )


@pytest.fixture
def geom_image():
    energy = np.logspace(-1.0, 1.0, 2)
    axis = MapAxis.from_edges(energy, name="energy", unit=u.TeV, interp="log")
    return WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), frame="galactic", axes=[axis]
    )


def get_exposure(geom_etrue):
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

    exposure_map = make_map_exposure_true_energy(
        pointing=SkyCoord(1, 0.5, unit="deg", frame="galactic"),
        livetime=1 * u.hr,
        aeff=aeff,
        geom=geom_etrue,
    )
    return exposure_map


def get_psf():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)
    psf_map = PSFMap.from_energy_dependent_table_psf(table_psf)
    return psf_map


@pytest.fixture
def sky_model():
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="test-model"
    )


def get_map_dataset(geom, geom_etrue, edisp="edispmap", name="test", **kwargs):
    """Returns a MapDatasets"""
    # define background model
    background = Map.from_geom(geom)
    background.data += 0.2

    psf = get_psf()
    exposure = get_exposure(geom_etrue)

    e_reco = geom.axes["energy"]
    e_true = geom_etrue.axes["energy_true"]

    if edisp == "edispmap":
        edisp = EDispMap.from_diagonal_response(energy_axis_true=e_true)
    elif edisp == "edispkernelmap":
        edisp = EDispKernelMap.from_diagonal_response(
            energy_axis=e_reco, energy_axis_true=e_true
        )
    elif edisp == "edispkernel":
        edisp = EDispKernel.from_diagonal_response(
            energy_true=e_true.edges, energy=e_reco.edges
        )
    else:
        edisp = None

    # define fit mask
    center = SkyCoord("0.2 deg", "0.1 deg", frame="galactic")
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    mask_fit = geom.region_mask([circle])
    mask_fit = Map.from_geom(geom, data=mask_fit)

    models = FoVBackgroundModel(dataset_name=name)

    return MapDataset(
        models=models,
        exposure=exposure,
        background=background,
        psf=psf,
        edisp=edisp,
        mask_fit=mask_fit,
        name=name,
        **kwargs,
    )


@requires_data()
def test_map_dataset_str(sky_model, geom, geom_etrue):
    dataset = get_map_dataset(geom, geom_etrue)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

    dataset.counts = dataset.npred()
    dataset.mask_safe = dataset.mask_fit
    assert "MapDataset" in str(dataset)
    assert "(frozen)" in str(dataset)
    assert "background" in str(dataset)

    dataset.mask_safe = None
    assert "MapDataset" in str(dataset)


@requires_data()
def test_fake(sky_model, geom, geom_etrue):
    """Test the fake dataset"""
    dataset = get_map_dataset(geom, geom_etrue)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

    npred = dataset.npred()
    assert np.all(npred.data >= 0)  # npred must be positive
    dataset.counts = npred
    real_dataset = dataset.copy()
    dataset.fake(314)

    assert real_dataset.counts.data.shape == dataset.counts.data.shape
    assert_allclose(real_dataset.counts.data.sum(), 9525.299054, rtol=1e-5)
    assert_allclose(dataset.counts.data.sum(), 9723)


@requires_data()
def test_different_exposure_unit(sky_model, geom):
    energy_range_true = np.logspace(2, 4, 3)
    axis = MapAxis.from_edges(
        energy_range_true, name="energy_true", unit="GeV", interp="log"
    )
    geom_gev = geom.to_image().to_cube([axis])
    dataset = get_map_dataset(geom, geom_gev, edisp="None")

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

    npred = dataset.npred()

    assert_allclose(npred.data[0, 50, 50], 6.086019)


@pytest.mark.parametrize(("edisp_mode"), ["edispmap", "edispkernelmap", "edispkernel"])
@requires_data()
def test_to_spectrum_dataset(sky_model, geom, geom_etrue, edisp_mode):

    dataset_ref = get_map_dataset(geom, geom_etrue, edisp=edisp_mode)

    bkg_model = FoVBackgroundModel(dataset_name=dataset_ref.name)
    dataset_ref.models = [sky_model, bkg_model]

    dataset_ref.counts = dataset_ref.npred_background() * 0.0
    dataset_ref.counts.data[1, 50, 50] = 1
    dataset_ref.counts.data[1, 60, 50] = 1

    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset_ref.gti = gti
    on_region = CircleSkyRegion(center=geom.center_skydir, radius=0.05 * u.deg)
    spectrum_dataset = dataset_ref.to_spectrum_dataset(on_region)
    spectrum_dataset_corrected = dataset_ref.to_spectrum_dataset(
        on_region, containment_correction=True
    )
    mask = np.ones_like(dataset_ref.counts, dtype="bool")
    mask[1, 40:60, 40:60] = 0
    dataset_ref.mask_safe = Map.from_geom(dataset_ref.counts.geom, data=mask)
    spectrum_dataset_mask = dataset_ref.to_spectrum_dataset(on_region)

    assert np.sum(spectrum_dataset.counts.data) == 1
    assert spectrum_dataset.data_shape == (2, 1, 1)
    assert spectrum_dataset.background.geom.axes[0].nbin == 2
    assert spectrum_dataset.exposure.geom.axes[0].nbin == 3
    assert spectrum_dataset.exposure.unit == "m2s"
    assert spectrum_dataset.edisp.get_edisp_kernel().energy_axis.nbin == 2
    assert spectrum_dataset.edisp.get_edisp_kernel().energy_axis_true.nbin == 3
    assert_allclose(spectrum_dataset.edisp.exposure_map.data[1], 3.070884e09, rtol=1e-5)
    assert np.sum(spectrum_dataset_mask.counts.data) == 0
    assert spectrum_dataset_mask.data_shape == (2, 1, 1)
    assert spectrum_dataset_corrected.exposure.unit == "m2s"
    assert_allclose(spectrum_dataset.exposure.data[1], 3.070884e09, rtol=1e-5)
    assert_allclose(spectrum_dataset_corrected.exposure.data[1], 2.035899e09, rtol=1e-5)


@requires_data()
def test_info_dict(sky_model, geom, geom_etrue):
    dataset = get_map_dataset(geom, geom_etrue)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

    dataset.counts = dataset.npred()
    info_dict = dataset.info_dict()

    assert_allclose(info_dict["counts"], 9526, rtol=1e-3)
    assert_allclose(info_dict["background"], 4000.0005, rtol=1e-3)
    assert_allclose(info_dict["npred_background"], 4000.0, rtol=1e-3)
    assert_allclose(info_dict["excess"], 5525.756, rtol=1e-3)
    assert_allclose(info_dict["exposure_min"].value, 8.32e8, rtol=1e-3)
    assert_allclose(info_dict["exposure_max"].value, 1.105e10, rtol=1e-3)
    assert info_dict["exposure_max"].unit == "m2 s"
    assert info_dict["name"] == "test"

    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset.gti = gti
    info_dict = dataset.info_dict()
    assert_allclose(info_dict["counts"], 9526, rtol=1e-3)
    assert_allclose(info_dict["background"], 4000.0005, rtol=1e-3)
    assert_allclose(info_dict["npred_background"], 4000.0, rtol=1e-3)
    assert_allclose(info_dict["sqrt_ts"], 74.024180, rtol=1e-3)
    assert_allclose(info_dict["excess"], 5525.756, rtol=1e-3)
    assert_allclose(info_dict["ontime"].value, 3600)

    assert info_dict["name"] == "test"


def get_fermi_3fhl_gc_dataset():
    counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz")
    background = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background-cube.fits.gz"
    )
    bkg_model = FoVBackgroundModel(dataset_name="fermi-3fhl-gc")

    exposure = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
    )
    return MapDataset(
        counts=counts,
        background=background,
        models=[bkg_model],
        exposure=exposure,
        name="fermi-3fhl-gc",
    )


@requires_data()
def test_resample_energy_3fhl():
    dataset = get_fermi_3fhl_gc_dataset()

    new_axis = MapAxis.from_edges([10, 35, 100] * u.GeV, interp="log", name="energy")
    grouped = dataset.resample_energy_axis(energy_axis=new_axis)

    assert grouped.counts.data.shape == (2, 200, 400)
    assert grouped.counts.data[0].sum() == 28581
    assert_allclose(
        grouped.npred_background().data.sum(axis=(1, 2)),
        [25074.366386, 2194.298612],
        rtol=1e-5,
    )
    assert_allclose(grouped.exposure.data, dataset.exposure.data, rtol=1e-5)

    axis = grouped.counts.geom.axes[0]
    npred = dataset.npred()
    npred_grouped = grouped.npred()
    assert_allclose(npred.resample_axis(axis=axis).data.sum(), npred_grouped.data.sum())


@requires_data()
def test_to_image_3fhl():
    dataset = get_fermi_3fhl_gc_dataset()

    dataset_im = dataset.to_image()

    assert dataset_im.counts.data.sum() == dataset.counts.data.sum()
    assert_allclose(dataset_im.npred_background().data.sum(), 28548.625, rtol=1e-5)
    assert_allclose(dataset_im.exposure.data, dataset.exposure.data, rtol=1e-5)

    npred = dataset.npred()
    npred_im = dataset_im.npred()
    assert_allclose(npred.data.sum(), npred_im.data.sum())


def test_to_image_mask_safe():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2)
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.5, width=(1, 1), frame="icrs", axes=[axis]
    )
    dataset = MapDataset.create(geom)

    # Check map_safe handling
    data = np.array([[[False, True], [True, True]], [[False, False], [True, True]]])
    dataset.mask_safe = WcsNDMap.from_geom(geom=geom, data=data)

    dataset_im = dataset.to_image()
    assert dataset_im.mask_safe.data.dtype == bool

    desired = np.array([[False, True], [True, True]])
    assert (dataset_im.mask_safe.data == desired).all()

    # Check that missing entries in the dataset do not break
    dataset_copy = dataset.copy()
    dataset_copy.exposure = None
    dataset_im = dataset_copy.to_image()
    assert dataset_im.exposure is None

    dataset_copy = dataset.copy()
    dataset_copy.counts = None
    dataset_im = dataset_copy.to_image()
    assert dataset_im.counts is None


@requires_data()
def test_downsample():
    dataset = get_fermi_3fhl_gc_dataset()

    downsampled = dataset.downsample(2)

    assert downsampled.counts.data.shape == (11, 100, 200)
    assert downsampled.counts.data.sum() == dataset.counts.data.sum()
    assert_allclose(
        downsampled.npred_background().data.sum(axis=(1, 2)),
        dataset.npred_background().data.sum(axis=(1, 2)),
        rtol=1e-5,
    )
    assert_allclose(downsampled.exposure.data[5, 50, 100], 3.318082e11, rtol=1e-5)

    with pytest.raises(ValueError):
        dataset.downsample(2, axis_name="energy")


@requires_data()
def test_map_dataset_fits_io(tmp_path, sky_model, geom, geom_etrue):
    dataset = get_map_dataset(geom, geom_etrue)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

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
        "EDISP",
        "EDISP_BANDS",
        "EDISP_EXPOSURE",
        "EDISP_EXPOSURE_BANDS",
        "PSF",
        "PSF_BANDS",
        "PSF_EXPOSURE",
        "PSF_EXPOSURE_BANDS",
        "MASK_SAFE",
        "MASK_SAFE_BANDS",
        "MASK_FIT",
        "MASK_FIT_BANDS",
        "GTI",
    ]

    assert actual == desired

    dataset.write(tmp_path / "test.fits")

    dataset_new = MapDataset.read(tmp_path / "test.fits")

    assert dataset_new.mask.data.dtype == bool

    assert_allclose(dataset.counts.data, dataset_new.counts.data)
    assert_allclose(
        dataset.npred_background().data, dataset_new.npred_background().data
    )
    assert_allclose(dataset.edisp.edisp_map.data, dataset_new.edisp.edisp_map.data)
    assert_allclose(dataset.psf.psf_map.data, dataset_new.psf.psf_map.data)
    assert_allclose(dataset.exposure.data, dataset_new.exposure.data)
    assert_allclose(dataset.mask_fit.data, dataset_new.mask_fit.data)
    assert_allclose(dataset.mask_safe.data, dataset_new.mask_safe.data)

    assert dataset.counts.geom == dataset_new.counts.geom
    assert dataset.exposure.geom == dataset_new.exposure.geom
    assert dataset.npred_background().geom == dataset_new.npred_background().geom
    assert dataset.edisp.edisp_map.geom == dataset_new.edisp.edisp_map.geom

    assert_allclose(
        dataset.gti.time_sum.to_value("s"), dataset_new.gti.time_sum.to_value("s")
    )

    # To test io of psf and edisp map
    stacked = MapDataset.create(geom)
    stacked.write(tmp_path / "test-2.fits", overwrite=True)
    stacked1 = MapDataset.read(tmp_path / "test-2.fits")
    assert stacked1.psf.psf_map is not None
    assert stacked1.psf.exposure_map is not None
    assert stacked1.edisp.edisp_map is not None
    assert stacked1.edisp.exposure_map is not None
    assert stacked.mask.data.dtype == bool

    assert_allclose(stacked1.psf.psf_map, stacked.psf.psf_map)
    assert_allclose(stacked1.edisp.edisp_map, stacked.edisp.edisp_map)


@requires_dependency("iminuit")
@requires_dependency("matplotlib")
@requires_data()
def test_map_fit(sky_model, geom, geom_etrue):
    dataset_1 = get_map_dataset(geom, geom_etrue, name="test-1")
    dataset_2 = get_map_dataset(geom, geom_etrue, name="test-2")
    datasets = Datasets([dataset_1, dataset_2])

    models = Models(datasets.models)
    models.insert(0, sky_model)

    models["test-1-bkg"].spectral_model.norm.value = 0.5
    models["test-model"].spatial_model.sigma.frozen = True

    datasets.models = models
    dataset_2.counts = dataset_2.npred()
    dataset_1.counts = dataset_1.npred()

    models["test-1-bkg"].spectral_model.norm.value = 0.49
    models["test-2-bkg"].spectral_model.norm.value = 0.99

    fit = Fit(datasets)
    result = fit.run()

    assert result.success
    assert "minuit" in repr(result)

    npred = dataset_1.npred().data.sum()
    assert_allclose(npred, 7525.790688, rtol=1e-3)
    assert_allclose(result.total_stat, 21659.2139, rtol=1e-3)

    pars = result.parameters
    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars["lon_0"].error, 0.002244, rtol=1e-2)

    assert_allclose(pars["index"].value, 3, rtol=1e-2)
    assert_allclose(pars["index"].error, 0.024277, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars["amplitude"].error, 4.216154e-13, rtol=1e-2)

    # background norm 1
    assert_allclose(pars[8].value, 0.5, rtol=1e-2)
    assert_allclose(pars[8].error, 0.015811, rtol=1e-2)

    # background norm 2
    assert_allclose(pars[11].value, 1, rtol=1e-2)
    assert_allclose(pars[11].error, 0.02147, rtol=1e-2)

    # test mask_safe evaluation
    mask_safe = geom.energy_mask(energy_min=1 * u.TeV)
    dataset_1.mask_safe = Map.from_geom(geom, data=mask_safe)
    dataset_2.mask_safe = Map.from_geom(geom, data=mask_safe)

    stat = fit.datasets.stat_sum()
    assert_allclose(stat, 14823.579908, rtol=1e-5)

    region = sky_model.spatial_model.to_region()

    initial_counts = dataset_1.counts.copy()
    with mpl_plot_check():
        dataset_1.plot_residuals(kwargs_spectral=dict(region=region))

    # check dataset has not changed
    assert initial_counts == dataset_1.counts

    # test model evaluation outside image
    dataset_1.models[0].spatial_model.lon_0.value = 150
    dataset_1.npred()
    assert not dataset_1._evaluators["test-model"].contributes


@requires_dependency("iminuit")
@requires_data()
def test_map_fit_one_energy_bin(sky_model, geom_image):
    energy_axis = geom_image.axes["energy"]
    geom_etrue = geom_image.to_image().to_cube([energy_axis.copy(name="energy_true")])

    dataset = get_map_dataset(geom_image, geom_etrue)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [sky_model, bkg_model]

    sky_model.spectral_model.index.value = 3.0
    sky_model.spectral_model.index.frozen = True
    dataset.models[f"{dataset.name}-bkg"].spectral_model.norm.value = 0.5

    dataset.counts = dataset.npred()

    # Move a bit away from the best-fit point, to make sure the optimiser runs
    sky_model.parameters["sigma"].value = 0.21
    dataset.models[f"{dataset.name}-bkg"].parameters["norm"].frozen = True

    fit = Fit([dataset])
    result = fit.run()

    assert result.success

    npred = dataset.npred().data.sum()
    assert_allclose(npred, 16538.124036, rtol=1e-3)
    assert_allclose(result.total_stat, -34844.125047, rtol=1e-3)

    pars = result.parameters

    assert_allclose(pars["lon_0"].value, 0.2, rtol=1e-2)
    assert_allclose(pars["lon_0"].error, 0.001689, rtol=1e-2)

    assert_allclose(pars["sigma"].value, 0.2, rtol=1e-2)
    assert_allclose(pars["sigma"].error, 0.00092, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e-11, rtol=1e-2)
    assert_allclose(pars["amplitude"].error, 8.127593e-14, rtol=1e-2)


def test_create():
    # tests empty datasets created
    rad_axis = MapAxis(nodes=np.linspace(0.0, 1.0, 51), unit="deg", name="rad")
    e_reco = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 3), name="energy", unit=u.TeV, interp="log"
    )
    e_true = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 4), name="energy_true", unit=u.TeV, interp="log"
    )
    geom = WcsGeom.create(binsz=0.02, width=(2, 2), axes=[e_reco])
    empty_dataset = MapDataset.create(
        geom=geom, energy_axis_true=e_true, rad_axis=rad_axis
    )

    assert empty_dataset.counts.data.shape == (2, 100, 100)

    assert empty_dataset.exposure.data.shape == (3, 100, 100)

    assert empty_dataset.psf.psf_map.data.shape == (3, 50, 10, 10)
    assert empty_dataset.psf.exposure_map.data.shape == (3, 1, 10, 10)

    assert isinstance(empty_dataset.edisp, EDispKernelMap)
    assert empty_dataset.edisp.edisp_map.data.shape == (3, 2, 10, 10)
    assert empty_dataset.edisp.exposure_map.data.shape == (3, 1, 10, 10)
    assert_allclose(empty_dataset.edisp.edisp_map.data.sum(), 300)

    assert_allclose(empty_dataset.gti.time_delta, 0.0 * u.s)


def test_create_with_migra(tmp_path):
    # tests empty datasets created
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")
    rad_axis = MapAxis(nodes=np.linspace(0.0, 1.0, 51), unit="deg", name="rad")
    e_reco = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 3), name="energy", unit=u.TeV, interp="log"
    )
    e_true = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 4), name="energy_true", unit=u.TeV, interp="log"
    )
    geom = WcsGeom.create(binsz=0.02, width=(2, 2), axes=[e_reco])
    empty_dataset = MapDataset.create(
        geom=geom, energy_axis_true=e_true, migra_axis=migra_axis, rad_axis=rad_axis
    )

    empty_dataset.write(tmp_path / "test.fits")

    dataset_new = MapDataset.read(tmp_path / "test.fits")

    assert isinstance(empty_dataset.edisp, EDispMap)
    assert empty_dataset.edisp.edisp_map.data.shape == (3, 50, 10, 10)
    assert empty_dataset.edisp.exposure_map.data.shape == (3, 1, 10, 10)
    assert_allclose(empty_dataset.edisp.edisp_map.data.sum(), 300)

    assert_allclose(empty_dataset.gti.time_delta, 0.0 * u.s)

    assert isinstance(dataset_new.edisp, EDispMap)
    assert dataset_new.edisp.edisp_map.data.shape == (3, 50, 10, 10)


def test_stack(sky_model):
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3)
    geom = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.05,
        width=(2, 2),
        frame="icrs",
        axes=[axis],
    )
    axis_etrue = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=5, name="energy_true"
    )
    geom_etrue = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.05,
        width=(2, 2),
        frame="icrs",
        axes=[axis_etrue],
    )

    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=axis, energy_axis_true=axis_etrue, geom=geom
    )
    edisp.exposure_map.quantity = (
        1e0 * u.m ** 2 * u.s * np.ones(edisp.exposure_map.data.shape)
    )

    bkg1 = Map.from_geom(geom)
    bkg1.data += 0.2

    cnt1 = Map.from_geom(geom)
    cnt1.data = 1.0 * np.ones(cnt1.data.shape)

    exp1 = Map.from_geom(geom_etrue)
    exp1.quantity = 1e7 * u.m ** 2 * u.s * np.ones(exp1.data.shape)

    mask1 = Map.from_geom(geom)
    mask1.data = np.ones(mask1.data.shape, dtype=bool)
    mask1.data[0][:][5:10] = False
    dataset1 = MapDataset(
        counts=cnt1,
        background=bkg1,
        exposure=exp1,
        mask_safe=mask1,
        name="dataset-1",
        edisp=edisp,
        meta_table=Table({"OBS_ID": [0]}),
    )

    bkg2 = Map.from_geom(geom)
    bkg2.data = 0.1 * np.ones(bkg2.data.shape)

    cnt2 = Map.from_geom(geom)
    cnt2.data = 1.0 * np.ones(cnt2.data.shape)

    exp2 = Map.from_geom(geom_etrue)
    exp2.quantity = 1e7 * u.m ** 2 * u.s * np.ones(exp2.data.shape)

    mask2 = Map.from_geom(geom)
    mask2.data = np.ones(mask2.data.shape, dtype=bool)
    mask2.data[0][:][5:10] = False
    mask2.data[1][:][10:15] = False

    dataset2 = MapDataset(
        counts=cnt2,
        background=bkg2,
        exposure=exp2,
        mask_safe=mask2,
        name="dataset-2",
        edisp=edisp,
        meta_table=Table({"OBS_ID": [1]}),
    )

    background_model2 = FoVBackgroundModel(dataset_name="dataset-2")
    background_model1 = FoVBackgroundModel(dataset_name="dataset-1")

    dataset1.models = [background_model1, sky_model]
    dataset2.models = [background_model2, sky_model]
    dataset1.stack(dataset2)

    dataset1.models = [sky_model]
    npred_b = dataset1.npred()

    assert_allclose(npred_b.data.sum(), 1459.985035, 1e-5)
    assert_allclose(dataset1.npred_background().data.sum(), 1360.00, 1e-5)
    assert_allclose(dataset1.counts.data.sum(), 9000, 1e-5)
    assert_allclose(dataset1.mask_safe.data.sum(), 4600)
    assert_allclose(dataset1.exposure.data.sum(), 1.6e11)

    assert_allclose(dataset1.meta_table["OBS_ID"][0], [0, 1])


@requires_data()
def test_npred_sig(sky_model, geom, geom_etrue):
    dataset = get_map_dataset(geom, geom_etrue)

    pwl = PowerLawSpectralModel()
    gauss = GaussianSpatialModel(
        lon_0="0.0 deg", lat_0="0.0 deg", sigma="0.5 deg", frame="galactic"
    )
    model1 = SkyModel(pwl, gauss)

    bkg = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [bkg, sky_model, model1]

    assert_allclose(dataset.npred().data.sum(), 9676.047906, rtol=1e-3)
    assert_allclose(dataset.npred_signal().data.sum(), 5676.04790, rtol=1e-3)
    assert_allclose(dataset.npred_signal(model=model1).data.sum(), 150.7487, rtol=1e-3)


def test_stack_npred():
    pwl = PowerLawSpectralModel()
    gauss = GaussianSpatialModel(sigma="0.2 deg")
    model = SkyModel(pwl, gauss)

    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=5)
    axis_etrue = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=11, name="energy_true"
    )

    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.05, width=(2, 2), frame="icrs", axes=[axis],
    )

    dataset_1 = MapDataset.create(
        geom,
        energy_axis_true=axis_etrue,
        name="dataset-1",
        gti=GTI.create("0 min", "30 min"),
    )
    dataset_1.psf = None
    dataset_1.exposure.data += 1
    dataset_1.mask_safe.data = geom.energy_mask(energy_min=1 * u.TeV)
    dataset_1.background.data += 1

    bkg_model_1 = FoVBackgroundModel(dataset_name=dataset_1.name)
    dataset_1.models = [model, bkg_model_1]

    dataset_2 = MapDataset.create(
        geom,
        energy_axis_true=axis_etrue,
        name="dataset-2",
        gti=GTI.create("30 min", "60 min"),
    )
    dataset_2.psf = None
    dataset_2.exposure.data += 1
    dataset_2.mask_safe.data = geom.energy_mask(energy_min=0.2 * u.TeV)
    dataset_2.background.data += 1

    bkg_model_2 = FoVBackgroundModel(dataset_name=dataset_2.name)
    dataset_2.models = [model, bkg_model_2]

    npred_1 = dataset_1.npred()
    npred_1.data[~dataset_1.mask_safe.data] = 0
    npred_2 = dataset_2.npred()
    npred_2.data[~dataset_2.mask_safe.data] = 0

    stacked_npred = Map.from_geom(geom)
    stacked_npred.stack(npred_1)
    stacked_npred.stack(npred_2)

    stacked = MapDataset.create(geom, energy_axis_true=axis_etrue, name="stacked")
    stacked.stack(dataset_1)
    stacked.stack(dataset_2)

    npred_stacked = stacked.npred()

    assert_allclose(npred_stacked.data, stacked_npred.data)


def to_cube(image):
    # introduce a fake enery axis for now
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy")
    geom = image.geom.to_cube([axis])
    return WcsNDMap.from_geom(geom=geom, data=image.data)


@pytest.fixture
def images():
    """Load some `counts`, `counts_off`, `acceptance_on`, `acceptance_off" images"""
    filename = "$GAMMAPY_DATA/tests/unbundled/hess/survey/hess_survey_snippet.fits.gz"
    return {
        "counts": to_cube(WcsNDMap.read(filename, hdu="ON")),
        "counts_off": to_cube(WcsNDMap.read(filename, hdu="OFF")),
        "acceptance": to_cube(WcsNDMap.read(filename, hdu="ONEXPOSURE")),
        "acceptance_off": to_cube(WcsNDMap.read(filename, hdu="OFFEXPOSURE")),
        "exposure": to_cube(WcsNDMap.read(filename, hdu="EXPGAMMAMAP")),
        "background": to_cube(WcsNDMap.read(filename, hdu="BACKGROUND")),
    }


def test_npred_psf_after_edisp():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.8 TeV", "15 TeV", nbin=6, name="energy_true"
    )

    geom = WcsGeom.create(width=4 * u.deg, binsz=0.02, axes=[energy_axis])
    dataset = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)
    dataset.background.data += 1
    dataset.exposure.data += 1e12
    dataset.mask_safe.data += True
    dataset.psf = PSFMap.from_gauss(
        energy_axis_true=energy_axis_true, sigma=0.2 * u.deg
    )

    model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        spatial_model=PointSpatialModel(),
        name="test-model",
    )

    model.apply_irf["psf_after_edisp"] = True

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [bkg_model, model]

    npred = dataset.npred()
    assert_allclose(npred.data.sum(), 129553.858658)


def get_map_dataset_onoff(images, **kwargs):
    """Returns a MapDatasetOnOff"""
    mask_geom = images["counts"].geom
    mask_data = np.ones(images["counts"].data.shape, dtype=bool)
    mask_safe = Map.from_geom(mask_geom, data=mask_data)
    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")

    return MapDatasetOnOff(
        counts=images["counts"],
        counts_off=images["counts_off"],
        acceptance=images["acceptance"],
        acceptance_off=images["acceptance_off"],
        exposure=images["exposure"],
        mask_safe=mask_safe,
        gti=gti,
        **kwargs,
    )


@requires_data()
def test_map_dataset_on_off_fits_io(images, tmp_path):
    dataset = get_map_dataset_onoff(images)
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
        "MASK_SAFE",
        "MASK_SAFE_BANDS",
        "GTI",
        "COUNTS_OFF",
        "COUNTS_OFF_BANDS",
        "ACCEPTANCE",
        "ACCEPTANCE_BANDS",
        "ACCEPTANCE_OFF",
        "ACCEPTANCE_OFF_BANDS",
    ]

    assert actual == desired

    dataset.write(tmp_path / "test.fits")

    dataset_new = MapDatasetOnOff.read(tmp_path / "test.fits")
    assert dataset_new.mask.data.dtype == bool

    assert_allclose(dataset.counts.data, dataset_new.counts.data)
    assert_allclose(dataset.counts_off.data, dataset_new.counts_off.data)
    assert_allclose(dataset.acceptance.data, dataset_new.acceptance.data)
    assert_allclose(dataset.acceptance_off.data, dataset_new.acceptance_off.data)
    assert_allclose(dataset.exposure.data, dataset_new.exposure.data)
    assert_allclose(dataset.mask_safe, dataset_new.mask_safe)

    assert np.all(dataset.mask_safe.data == dataset_new.mask_safe.data) == True
    assert dataset.mask_safe.geom == dataset_new.mask_safe.geom
    assert dataset.counts.geom == dataset_new.counts.geom
    assert dataset.exposure.geom == dataset_new.exposure.geom

    assert_allclose(
        dataset.gti.time_sum.to_value("s"), dataset_new.gti.time_sum.to_value("s")
    )


def test_create_onoff(geom):
    # tests empty datasets created

    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")
    rad_axis = MapAxis(nodes=np.linspace(0.0, 1.0, 51), unit="deg", name="rad")
    energy_axis = geom.axes["energy"].copy(name="energy_true")

    empty_dataset = MapDatasetOnOff.create(geom, energy_axis, migra_axis, rad_axis)

    assert_allclose(empty_dataset.counts.data.sum(), 0.0)
    assert_allclose(empty_dataset.counts_off.data.sum(), 0.0)
    assert_allclose(empty_dataset.acceptance.data.sum(), 0.0)
    assert_allclose(empty_dataset.acceptance_off.data.sum(), 0.0)

    assert empty_dataset.psf.psf_map.data.shape == (2, 50, 10, 10)
    assert empty_dataset.psf.exposure_map.data.shape == (2, 1, 10, 10)

    assert empty_dataset.edisp.edisp_map.data.shape == (2, 50, 10, 10)
    assert empty_dataset.edisp.exposure_map.data.shape == (2, 1, 10, 10)

    assert_allclose(empty_dataset.edisp.edisp_map.data.sum(), 200)

    assert_allclose(empty_dataset.gti.time_delta, 0.0 * u.s)


@requires_data()
def test_map_dataset_onoff_str(images):
    dataset = get_map_dataset_onoff(images)
    assert "MapDatasetOnOff" in str(dataset)


@requires_data()
def test_stack_onoff(images):
    dataset = get_map_dataset_onoff(images)
    stacked = dataset.copy()

    stacked.stack(dataset)

    assert_allclose(stacked.counts.data.sum(), 2 * dataset.counts.data.sum())
    assert_allclose(stacked.counts_off.data.sum(), 2 * dataset.counts_off.data.sum())
    assert_allclose(
        stacked.acceptance.data.sum(), dataset.data_shape[1] * dataset.data_shape[2]
    )
    assert_allclose(
        np.nansum(stacked.acceptance_off.data),
        np.nansum(
            dataset.counts_off.data / (dataset.counts_off.data * dataset.alpha.data)
        ),
    )
    assert_allclose(stacked.exposure.data, 2.0 * dataset.exposure.data)


def test_dataset_cutout_aligned(geom):
    dataset = MapDataset.create(geom)

    kwargs = {"position": geom.center_skydir, "width": 1 * u.deg}
    geoms = {name: geom.cutout(**kwargs) for name, geom in dataset.geoms.items()}

    cutout = MapDataset.from_geoms(**geoms, name="cutout")

    assert dataset.counts.geom.is_aligned(cutout.counts.geom)
    assert dataset.exposure.geom.is_aligned(cutout.exposure.geom)
    assert dataset.edisp.edisp_map.geom.is_aligned(cutout.edisp.edisp_map.geom)
    assert dataset.psf.psf_map.geom.is_aligned(cutout.psf.psf_map.geom)


def test_stack_onoff_cutout(geom_image):
    # Test stacking of cutouts
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=3, name="energy_true"
    )

    dataset = MapDatasetOnOff.create(geom_image, energy_axis_true=energy_axis_true)
    dataset.gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")

    kwargs = {"position": geom_image.center_skydir, "width": 1 * u.deg}
    geoms = {name: geom.cutout(**kwargs) for name, geom in dataset.geoms.items()}

    dataset_cutout = MapDatasetOnOff.from_geoms(**geoms, name="cutout-dataset")
    dataset_cutout.gti = GTI.create(
        [0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00"
    )
    dataset_cutout.mask_safe.data += True
    dataset_cutout.counts.data += 1
    dataset_cutout.counts_off.data += 1
    dataset_cutout.exposure.data += 1

    dataset.stack(dataset_cutout)

    assert_allclose(dataset.counts.data.sum(), 2500)
    assert_allclose(dataset.counts_off.data.sum(), 2500)
    assert_allclose(dataset.alpha.data.sum(), 0)
    assert_allclose(dataset.exposure.data.sum(), 7500)
    assert dataset_cutout.name == "cutout-dataset"


def test_datasets_io_no_model(tmpdir):
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=2)
    geom = WcsGeom.create(npix=(5, 5), axes=[axis])
    dataset_1 = MapDataset.create(geom, name="dataset_1")
    dataset_2 = MapDataset.create(geom, name="dataset_2")

    datasets = Datasets([dataset_1, dataset_2])

    datasets.write(filename=tmpdir / "datasets.yaml")

    filename_1 = tmpdir / "dataset_1.fits"
    assert filename_1.exists()

    filename_2 = tmpdir / "dataset_2.fits"
    assert filename_2.exists()


@requires_data()
def test_map_dataset_on_off_to_spectrum_dataset(images):
    dataset = get_map_dataset_onoff(images)

    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset.gti = gti

    on_region = CircleSkyRegion(
        center=dataset.counts.geom.center_skydir, radius=0.1 * u.deg
    )

    spectrum_dataset = dataset.to_spectrum_dataset(on_region)

    assert spectrum_dataset.counts.data[0] == 8
    assert spectrum_dataset.data_shape == (1, 1, 1)
    assert spectrum_dataset.counts_off.data[0] == 33914
    assert_allclose(spectrum_dataset.alpha.data[0], 0.0002143, atol=1e-7)

    excess_map = images["counts"] - images["background"]
    excess_true = excess_map.get_spectrum(on_region, np.sum).data[0]

    excess = spectrum_dataset.excess.data[0]
    assert_allclose(excess, excess_true, rtol=1e-3)

    assert spectrum_dataset.name != dataset.name


@requires_data()
def test_map_dataset_on_off_to_spectrum_dataset_weights():
    e_reco = MapAxis.from_bounds(1, 10, nbin=3, unit="TeV", name="energy")

    geom = WcsGeom.create(
        skydir=(0, 0), width=(2.5, 2.5), binsz=0.5, axes=[e_reco], frame="galactic"
    )
    counts = Map.from_geom(geom)
    counts.data += 1
    counts_off = Map.from_geom(geom)
    counts_off.data += 2
    acceptance = Map.from_geom(geom)
    acceptance.data += 1
    acceptance_off = Map.from_geom(geom)
    acceptance_off.data += 4

    weights = Map.from_geom(geom, dtype="bool")
    weights.data[1:, 2:4, 2] = True

    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
        mask_safe=weights,
        gti=gti,
    )

    on_region = CircleSkyRegion(
        center=dataset.counts.geom.center_skydir, radius=1.5 * u.deg
    )

    spectrum_dataset = dataset.to_spectrum_dataset(on_region)

    assert_allclose(spectrum_dataset.counts.data[:, 0, 0], [0, 2, 2])
    assert_allclose(spectrum_dataset.counts_off.data[:, 0, 0], [0, 4, 4])
    assert_allclose(spectrum_dataset.acceptance.data[:, 0, 0], [0, 0.08, 0.08])
    assert_allclose(spectrum_dataset.acceptance_off.data[:, 0, 0], [0, 0.32, 0.32])
    assert_allclose(spectrum_dataset.alpha.data[:, 0, 0], [0, 0.25, 0.25])


@requires_data()
def test_map_dataset_on_off_cutout(images):
    dataset = get_map_dataset_onoff(images)
    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")
    dataset.gti = gti

    cutout_dataset = dataset.cutout(
        images["counts"].geom.center_skydir, ["1 deg", "1 deg"]
    )

    assert cutout_dataset.counts.data.shape == (1, 50, 50)
    assert cutout_dataset.counts_off.data.shape == (1, 50, 50)
    assert cutout_dataset.acceptance.data.shape == (1, 50, 50)
    assert cutout_dataset.acceptance_off.data.shape == (1, 50, 50)
    assert cutout_dataset.name != dataset.name


def test_map_dataset_on_off_fake(geom):
    rad_axis = MapAxis(nodes=np.linspace(0.0, 1.0, 51), unit="deg", name="rad")
    energy_true_axis = geom.axes["energy"].copy(name="energy_true")

    empty_dataset = MapDataset.create(geom, energy_true_axis, rad_axis=rad_axis)
    empty_dataset = MapDatasetOnOff.from_map_dataset(
        empty_dataset, acceptance=1, acceptance_off=10.0
    )

    empty_dataset.acceptance_off.data[0, 50, 50] = 0
    background_map = Map.from_geom(geom, data=1)
    empty_dataset.fake(background_map, random_state=42)

    assert_allclose(empty_dataset.counts.data[0, 50, 50], 0)
    assert_allclose(empty_dataset.counts.data.mean(), 0.99445, rtol=1e-3)
    assert_allclose(empty_dataset.counts_off.data.mean(), 10.00055, rtol=1e-3)


@requires_data()
def test_map_dataset_on_off_to_image():
    axis = MapAxis.from_energy_bounds(1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=(10, 10), binsz=0.05, axes=[axis])

    counts = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    counts_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off *= 2

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )
    image_dataset = dataset.to_image()

    assert image_dataset.counts.data.shape == (1, 10, 10)
    assert image_dataset.acceptance_off.data.shape == (1, 10, 10)
    assert_allclose(image_dataset.acceptance, 2)
    assert_allclose(image_dataset.acceptance_off, 4)
    assert_allclose(image_dataset.counts_off, 2)
    assert image_dataset.name != dataset.name

    # Try with a safe_mask
    mask_safe = Map.from_geom(geom, data=np.ones((2, 10, 10), dtype="bool"))
    mask_safe.data[0] = 0
    dataset.mask_safe = mask_safe
    image_dataset = dataset.to_image()

    assert_allclose(image_dataset.acceptance, 1)
    assert_allclose(image_dataset.acceptance_off, 2)
    assert_allclose(image_dataset.counts_off, 1)


def test_map_dataset_geom(geom, sky_model):
    e_true = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5, name="energy_true")
    dataset = MapDataset.create(geom, energy_axis_true=e_true)
    dataset.counts = None
    dataset.background = None

    npred = dataset.npred()
    assert npred.geom == geom

    dataset.mask_safe = None
    dataset.mask_fit = None

    with pytest.raises(ValueError):
        dataset._geom


@requires_data()
def test_names(geom, geom_etrue, sky_model):
    m = Map.from_geom(geom)
    m.quantity = 0.2 * np.ones(m.data.shape)
    background_model1 = FoVBackgroundModel(dataset_name="test")
    assert background_model1.name == "test-bkg"

    c_map1 = Map.from_geom(geom)
    c_map1.quantity = 0.3 * np.ones(c_map1.data.shape)

    model1 = sky_model.copy()
    assert model1.name != sky_model.name
    model1 = sky_model.copy(name="model1")
    assert model1.name == "model1"
    model2 = sky_model.copy(name="model2")

    dataset1 = MapDataset(
        counts=c_map1,
        models=Models([model1, model2, background_model1]),
        exposure=get_exposure(geom_etrue),
        background=m,
        name="test",
    )

    dataset2 = dataset1.copy()
    print(dataset2.models)
    assert dataset2.name != dataset1.name
    assert dataset2.models[f"{dataset2.name}-bkg"]
    dataset2 = dataset1.copy(name="dataset2")

    assert dataset2.name == "dataset2"
    assert dataset2.models["dataset2-bkg"].name == "dataset2-bkg"
    assert dataset2.models["dataset2-bkg"] is not dataset1.models["test-bkg"]
    assert dataset2.models.names == ["model1", "model2", "dataset2-bkg"]
    assert dataset2.models is not dataset1.models


def test_stack_dataset_dataset_on_off():
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy")
    geom = WcsGeom.create(width=1, axes=[axis])

    gti = GTI.create([0 * u.s], [1 * u.h])

    dataset = MapDataset.create(geom, gti=gti)
    dataset_on_off = MapDatasetOnOff.create(geom, gti=gti)
    dataset_on_off.mask_safe.data += True

    dataset_on_off.acceptance_off += 5
    dataset_on_off.acceptance += 1
    dataset_on_off.counts_off += 1
    dataset.stack(dataset_on_off)

    assert_allclose(dataset.npred_background().data, 0.166667, rtol=1e-3)


@requires_data()
def test_info_dict_on_off(images):
    dataset = get_map_dataset_onoff(images)
    info_dict = dataset.info_dict()
    assert_allclose(info_dict["counts"], 4299, rtol=1e-3)
    assert_allclose(info_dict["excess"], -22.52295, rtol=1e-3)
    assert_allclose(info_dict["exposure_min"].value, 1.739467e08, rtol=1e-3)
    assert_allclose(info_dict["exposure_max"].value, 3.4298378e09, rtol=1e-3)
    assert_allclose(info_dict["npred"], 4321.518, rtol=1e-3)
    assert_allclose(info_dict["counts_off"], 20407510.0, rtol=1e-3)
    assert_allclose(info_dict["acceptance"], 4272.7075, rtol=1e-3)
    assert_allclose(info_dict["acceptance_off"], 20175596.0, rtol=1e-3)
    assert_allclose(info_dict["alpha"], 0.000169, rtol=1e-3)
    assert_allclose(info_dict["ontime"].value, 3600)


def test_slice_by_idx():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=17)
    axis_etrue = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=31, name="energy_true"
    )

    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.5, width=(2, 2), frame="icrs", axes=[axis],
    )
    dataset = MapDataset.create(geom=geom, energy_axis_true=axis_etrue, binsz_irf=0.5)

    slices = {"energy": slice(5, 10)}
    sub_dataset = dataset.slice_by_idx(slices)

    assert sub_dataset.counts.geom.data_shape == (5, 4, 4)
    assert sub_dataset.mask_safe.geom.data_shape == (5, 4, 4)
    assert sub_dataset.npred_background().geom.data_shape == (5, 4, 4)
    assert sub_dataset.exposure.geom.data_shape == (31, 4, 4)
    assert sub_dataset.edisp.edisp_map.geom.data_shape == (31, 5, 4, 4)
    assert sub_dataset.psf.psf_map.geom.data_shape == (31, 66, 4, 4)

    axis = sub_dataset.counts.geom.axes["energy"]
    assert_allclose(axis.edges[0].value, 0.387468, rtol=1e-5)

    slices = {"energy_true": slice(5, 10)}
    sub_dataset = dataset.slice_by_idx(slices)

    assert sub_dataset.counts.geom.data_shape == (17, 4, 4)
    assert sub_dataset.mask_safe.geom.data_shape == (17, 4, 4)
    assert sub_dataset.npred_background().geom.data_shape == (17, 4, 4)
    assert sub_dataset.exposure.geom.data_shape == (5, 4, 4)
    assert sub_dataset.edisp.edisp_map.geom.data_shape == (5, 17, 4, 4)
    assert sub_dataset.psf.psf_map.geom.data_shape == (5, 66, 4, 4)

    axis = sub_dataset.counts.geom.axes["energy"]
    assert_allclose(axis.edges[0].value, 0.1, rtol=1e-5)

    axis = sub_dataset.exposure.geom.axes["energy_true"]
    assert_allclose(axis.edges[0].value, 0.210175, rtol=1e-5)


@requires_dependency("matplotlib")
def test_plot_residual_onoff():
    axis = MapAxis.from_energy_bounds(1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=(10, 10), binsz=0.05, axes=[axis])

    counts = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    counts_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off *= 2

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )
    with mpl_plot_check():
        dataset.plot_residuals_spatial()


def test_to_map_dataset():
    axis = MapAxis.from_energy_bounds(1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=(10, 10), binsz=0.05, axes=[axis])

    counts = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    counts_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off = Map.from_geom(geom, data=np.ones((2, 10, 10)))
    acceptance_off *= 2

    dataset_onoff = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )

    dataset = dataset_onoff.to_map_dataset(name="ds")

    assert dataset.name == "ds"
    assert_allclose(dataset.npred_background().data.sum(), 100)
    assert isinstance(dataset, MapDataset)
    assert dataset.counts == dataset_onoff.counts


def test_downsample_onoff():
    axis = MapAxis.from_energy_bounds(1, 10, 4, unit="TeV")
    geom = WcsGeom.create(npix=(10, 10), binsz=0.05, axes=[axis])

    counts = Map.from_geom(geom, data=np.ones((4, 10, 10)))
    counts_off = Map.from_geom(geom, data=np.ones((4, 10, 10)))
    acceptance = Map.from_geom(geom, data=np.ones((4, 10, 10)))
    acceptance_off = Map.from_geom(geom, data=np.ones((4, 10, 10)))
    acceptance_off *= 2

    dataset_onoff = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )

    downsampled = dataset_onoff.downsample(2, axis_name="energy")

    assert downsampled.counts.data.shape == (2, 10, 10)
    assert downsampled.counts.data.sum() == dataset_onoff.counts.data.sum()
    assert downsampled.counts_off.data.sum() == dataset_onoff.counts_off.data.sum()
    assert_allclose(downsampled.alpha.data, 0.5)
